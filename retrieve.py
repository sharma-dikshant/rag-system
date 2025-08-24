import chromadb
from chromadb.utils import embedding_functions
from config import GEMINI_API_KEY

# Reconnect to the existing DB
client = chromadb.PersistentClient(path="./chroma_db")

# Same embedding function as in ingest.py
embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GEMINI_API_KEY,
    model_name="models/embedding-001"
)

# Get the collection
collection = client.get_collection(
    "mini_rag", embedding_function=embedding_function)

# === Query ===
user_query = "What is the size of sun?"

results = collection.query(
    query_texts=[user_query],
    n_results=3,   # how many chunks to fetch
    include=["documents", "metadatas", "distances"]
)

print("Query:", user_query)
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"- {doc[:80]}... (score: {dist:.4f})")
