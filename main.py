from src import HybridSearchEngine

# we have already saved the vector store so that we donot need to add the data again and again to save time.
RETRIEVER = HybridSearchEngine(db_path="./vector_store/chroma_db")

def main():
    queries=[
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting."
    ]

    for query in queries:
        print(f"\n" + "="*80)
        print(f"SEARCHING: '{query}'")
        print("="*80)
        
        results=RETRIEVER.search(query, top_k=3)

        if not results:
            print("No results found for this query.")
        else:
            for score, meta in results:
                print(f"\nMatch | Score: {score:.4f} | Path: {meta['image_path']}")
                RETRIEVER.display_image(meta['image_path'])


if __name__ == "__main__":
    main()