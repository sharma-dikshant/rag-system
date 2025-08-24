from google import generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from config import GEMINI_API_KEY

# 1) Configure Gemini embeddings
genai.configure(api_key=GEMINI_API_KEY)
embedding_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GEMINI_API_KEY,
    model_name="models/embedding-001"   # Gemini embedding model
)

# 2) Simple doc loader
def load_docs(folder="docs"):
    texts, ids = [], []
    for p in Path(folder).glob("*.txt"):
        text = p.read_text(encoding="utf-8").strip()
        if text:
            texts.append(text)
            ids.append(p.stem)
    return ids, texts

# 3) Chunk text into smaller pieces
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def chunk_all(ids, texts):
    new_ids, chunks = [], []
    for doc_id, text in zip(ids, texts):
        parts = chunk_text(text)
        for idx, part in enumerate(parts):
            new_ids.append(f"{doc_id}_chunk{idx}")
            chunks.append(part)
    return new_ids, chunks

# 4) Build Chroma collection
def build_chroma(db_path="./chroma_db", collection_name="mini_rag"):
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
    )
    ids, texts = load_docs()
    ids, chunks = chunk_all(ids, texts)

    # 👉 Generate embeddings and print them
    # for i, chunk in zip(ids, chunks):
    #     vector = embedding_fn([chunk])  # returns [[vector]]
    #     print(f"\nID: {i}")
    #     print(f"Text: {chunk[:80]}...")  # print preview of text
    #     print(f"Vector length: {len(vector[0])}")
    #     print(f"Vector: {vector[0][:10]} ...\n")  # print only first 10 dims for readability

    # Store in Chroma
    col.upsert(ids=ids, documents=chunks)
    print(f"✅ Indexed {len(chunks)} chunks into '{collection_name}'")

if __name__ == "__main__":
    build_chroma()
