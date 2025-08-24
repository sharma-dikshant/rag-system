import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from config import GEMINI_API_KEY

# === Setup ===
genai.configure(api_key=GEMINI_API_KEY)

# Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

# Embedding function
embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GEMINI_API_KEY,
    model_name="models/embedding-001"
)

# Connect to the collection
collection = client.get_collection("mini_rag", embedding_function=embedding_function)

# === RAG function ===
def rag_answer(query: str, top_k: int = 3):
    # Step 1: Retrieve relevant chunks
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents"]
    )
    retrieved_chunks = results["documents"][0]

    # Step 2: Build strict prompt
    context = "\n".join(retrieved_chunks)
    prompt = f"""
    You are a helpful assistant.
    Use ONLY the information from the context below to answer the question.
    If the answer is not in the context, reply with: "I don’t know."

    Context:
    {context}

    Question: {query}

    Answer:
    """

    # Step 3: Generate answer using Gemini
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text


# === Try it ===
if __name__ == "__main__":
    user_query = "what is the size of sun"
    answer = rag_answer(user_query)
    print("Q:", user_query)
    print("A:", answer)
