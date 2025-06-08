import os
import openai
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

INDEX_PATH = "kb_index.faiss"
DOCS_PATH = "kb_docs.pkl"

# Load or initialize FAISS
if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        doc_texts = pickle.load(f)
else:
    index = faiss.IndexFlatL2(1536)
    doc_texts = []

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding
