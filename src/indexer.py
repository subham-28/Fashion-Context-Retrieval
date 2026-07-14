import os
import torch
import chromadb
from PIL import Image
from tqdm import tqdm
import open_clip
from transformers import BlipProcessor, BlipForQuestionAnswering


class FashionIndexer:
    def __init__(self, db_path, collection_name="hybrid_fashion_index", device=None):
        # device
        self.device=device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # BLIP model for generating JSON dictionary
        print(f"Initializing models on {self.device}...")
        self.vlm_processor=BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        self.vlm_model=BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(self.device)
        
        # fashionSigLIP model for generating image embeddings
        self.marqo_model,_,self.preprocess=open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
        self.marqo_model.eval().to(self.device)
        
        # Initialize Database
        self.client=chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
        
        # JSON template
        self.vqa_queries={
            "garment_top": "what is the type of upper clothing?",
            "color_top": "what is the primary color of the upper clothing?",
            "garment_bottom": "what is the type of lower clothing?",
            "color_bottom": "what is the primary color of the lower clothing?",
            "environment": "what is the specific environment or setting backdrop?",
            "style": "is the clothing style formal, casual, or streetwear?",
            "action": "what action is the person performing, such as sitting, standing, or walking?"
        }



    # this function makes a list of all img files
    def _get_image_paths(self, directory):
        valid_exts=('.jpg', '.jpeg', '.png', '.webp')
        return [os.path.join(root, f) for root, _, files in os.walk(directory) 
                for f in files if f.lower().endswith(valid_exts)]



    # this func store the data in the chroma db
    def run_indexing(self, image_dir, batch_size=32):
        image_paths=self._get_image_paths(image_dir)
        print(f"Found {len(image_paths)} images. Starting indexing...")

        batch_imgs,batch_metas,batch_embs,batch_ids=[], [], [], []

        for i,img_path in enumerate(tqdm(image_paths)):
            try:
                # open img and converted to RGB if incase its not
                raw_image=Image.open(img_path).convert('RGB')
                
                # Metadata Extraction - stores all the json answers and the img path
                meta = {}
                for key,query in self.vqa_queries.items():
                    inputs = self.vlm_processor(raw_image, query, return_tensors="pt").to(self.device)
                    with torch.inference_mode():
                        out=self.vlm_model.generate(**inputs, max_new_tokens=15)
                    ans=self.vlm_processor.decode(out[0], skip_special_tokens=True).strip()
                    meta[key]=ans if ans else "none"
                meta["image_path"]=img_path
                
                # Embedding - generated img embedding
                tensor=self.preprocess(raw_image).unsqueeze(0).to(self.device)
                with torch.inference_mode(),torch.cuda.amp.autocast():
                    emb=self.marqo_model.encode_image(tensor, normalize=True).cpu().tolist()[0]
                
                # Add to batch
                batch_imgs.append(img_path)
                batch_metas.append(meta)
                batch_embs.append(emb)
                batch_ids.append(f"img_{i}")

                # Batch Commit
                if len(batch_ids)>=batch_size:
                    self.collection.add(embeddings=batch_embs, metadatas=batch_metas, ids=batch_ids)
                    batch_imgs, batch_metas, batch_embs, batch_ids=[], [], [], []

            except Exception as e:
                print(f"Skipping {img_path}: {e}")

        # Final commit for remaining items
        if batch_ids:
            self.collection.add(embeddings=batch_embs, metadatas=batch_metas, ids=batch_ids)
        print("Indexing Complete.")



# if __name__ == "__main__":
#     indexer = FashionIndexer(db_path="./vector_store/chroma_db")
#     indexer.run_indexing(image_dir="./data/fashion_dataset")