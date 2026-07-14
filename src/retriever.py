import os
import re
import json
import torch
import chromadb
from PIL import Image
from IPython.display import display
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip


class HybridSearchEngine:
    def __init__(self, db_path, collection_name="hybrid_fashion_index", device=None):
        # cuda loaded
        self.device=device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Qwen Model for converting query into hard JSON filters
        self.parser_tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        self.parser_model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16, device_map="auto")

        # Fashion Model for converting query into a dense vector
        self.marqo_model, _, _=open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
        self.tokenizer=open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionSigLIP')
        self.marqo_model.eval().to(self.device)

        # Cross Encoder for semantic search
        self.reranker=CrossEncoder("BAAI/bge-reranker-base")

        # Load DB
        self.client=chromadb.PersistentClient(path=db_path)
        self.collection=self.client.get_collection(name=collection_name)



    # this function converts the query into structured json format
    def _parse_query(self, raw_query):
        prompt=f"""You are a search query metadata extractor. Analyze the user's fashion search query and extract the specific details into a strictly formatted JSON object. 
Only extract information explicitly mentioned. If a detail is not mentioned, use the value "none". Ignore accessories like ties, hats, or bags; focus only on the main garments.
Keys to extract: "garment_top", "color_top", "garment_bottom", "color_bottom", "environment", "action".

Example 1: Query: "Someone wearing a blue shirt sitting on a park bench."
JSON output: {{"garment_top": "shirt", "color_top": "blue", "garment_bottom": "none", "color_bottom": "none", "environment": "park", "action": "sitting"}}
Example 2: Query: "Professional business attire inside a modern office with black pants."
JSON output: {{"garment_top": "none", "color_top": "none", "garment_bottom": "pants", "color_bottom": "black", "environment": "office", "action": "none"}}
Example 3: Query: "A red tie and a white shirt in a formal setting."
JSON output: {{"garment_top": "shirt", "color_top": "white", "garment_bottom": "none", "color_bottom": "none", "environment": "formal", "action": "none"}}

Query: "{raw_query}"
JSON output:"""
        
        messages=[{"role": "system", "content": "You output only valid JSON dictionaries without markdown formatting."}, {"role": "user", "content": prompt}]
        text =self.parser_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs=self.parser_tokenizer([text], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs=self.parser_model.generate(**inputs, max_new_tokens=100, temperature=0.1)
        response=self.parser_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            # we are using regex filtering incase the llm voilated the json structure
            json_match=re.search(r'\{.*\}', response.strip(), re.DOTALL)
            return {k: v.lower() for k, v in json.loads(json_match.group(0)).items() if v.lower() != "none"} if json_match else {}
        except: return {}



    # this func retrive the valid images
    def search(self, raw_query, top_k=3):
        filters=self._parse_query(raw_query) # json is generated for filtering

        # Mathematically, we define our search space S as:
        # S = { i | ∀ (k,v) ∈ F, metadata(i)[k] = v }
        # where F is the set of key-value pairs (filters) and i is an image entry.
        where_clause=None
        if filters:
            if len(filters)==1:
                key, val = list(filters.items())[0] # Single constraint: Filter by image where metadata[k] == v
                where_clause={key: {"$eq": val}}
            else:
                where_clause={"$and": [{k: {"$eq": v}} for k, v in filters.items()]} # Multi-constraint intersection: Filter by image where metadata[k1] == v1 AND metadata[k2] == v2 ...
        
        # Convert the user query into a dense vector embedding using the multimodal model
        # Let q be the query and V_q be its embedding vector: V_q = f_text(q) 
        # We normalize the vector such that ||V_q|| = 1 to perform cosine similarity.
        text_tokens=self.tokenizer([raw_query]).to(self.device)
        with torch.inference_mode():
            if self.device=="cuda":
                with torch.cuda.amp.autocast():
                    query_emb=self.marqo_model.encode_text(text_tokens, normalize=True).cpu().tolist()[0]
            else:
                query_emb=self.marqo_model.encode_text(text_tokens, normalize=True).cpu().tolist()[0]
        
        results=self.collection.query(query_embeddings=[query_emb], n_results=30, where=where_clause) # Perform initial retrieval from the database based on vector similarity
        
        if not results['metadatas'][0]: # If metadata filters are too restrictive, perform a broad semantic search
            results=self.collection.query(query_embeddings=[query_emb], n_results=20)
            
        metadatas=results['metadatas'][0]
        if not metadatas: return []
        
        # similarity scores are generated
        pairs=[[raw_query, f"Action: {m.get('action')}. Env: {m.get('environment')}. Upper: {m.get('color_top')} {m.get('garment_top')}. Lower: {m.get('color_bottom')} {m.get('garment_bottom')}."] for m in metadatas]
        scores=self.reranker.predict(pairs)

        # Sort candidates by their reranking score in descending orde
        scored=sorted(zip(scores, metadatas), key=lambda x: x[0], reverse=True)
        return scored[:top_k]
    

    
    def display_image(self, img_path):
        filename=os.path.basename(img_path)
        
        # C:\ML-DL\Projects\Fashion-Search\data\fashion_dataset\filename.jpg
        local_path=os.path.join(
            r"C:\ML-DL\Projects\Fashion-Search\data\fashion_dataset", 
            filename
        )
        
        if os.path.exists(local_path):
            img=Image.open(local_path)
            img.thumbnail((300, 300))
            img.show() 
            print(f"Opened: {filename}")
        else:
            print(f"File not found at: {local_path}")

# if __name__ == "__main__":
#     engine = HybridSearchEngine(db_path="/content/chroma_db")
#     results = engine.search("A person in a bright yellow raincoat.")
#     for score, meta in results:
#         print(f"Score: {score:.4f} | Path: {meta['image_path']}")