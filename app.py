from google import generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from config import GEMINI_API_KEY


# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

resp = model.generate_content("Hello, how can I help you today?")
print(resp.text)
