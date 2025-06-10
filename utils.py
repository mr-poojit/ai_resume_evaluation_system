import os
import openai
import faiss
import pickle
import numpy as np
import re
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

INDEX_PATH = "kb_index.faiss"
DOCS_PATH = "kb_docs.pkl"

def detect_intent_and_route(query: str):
    query = re.sub(r"^(hi|hello|hey)[, ]*", "", query.lower().strip())
    
    if "generate jd" in query or "job description" in query:
        return "generate_jd"
    elif "parse resume" in query:
        return "parse_resume"
    elif "match resumes" in query or "compare resume" in query:
        return "match_resumes"
    elif "find duplicate" in query:
        return "find_duplicates"
    elif "job embed" in query:
        return "job_embed"
    else:
        return "chat"

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
