import chromadb

def show_vectors(db_path="./chroma_db", collection_name="mini_rag"):
    # connect to existing DB
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_collection(name=collection_name)

    # get all items back
    results = col.get(include=["embeddings", "documents", "metadatas"])

    # print embeddings
    for idx, (doc, emb) in enumerate(zip(results["documents"], results["embeddings"])):
        print(f"\n--- Document {idx+1} ---")
        print(f"Text: {doc[:80]}...")  # preview text
        print(f"Vector length: {len(emb)}")
        print(f"First 10 values: {emb[:10]} ...")

if __name__ == "__main__":
    show_vectors()
