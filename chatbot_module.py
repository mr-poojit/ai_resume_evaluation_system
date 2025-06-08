from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
import os
import openai
import faiss
import pickle
import numpy as np
from models import SessionLocal, ChatHistory, ManualProblem
from utils import get_embedding, index, doc_texts 
from dotenv import load_dotenv
from datetime import datetime
import tempfile

# Load OpenAI Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

router = APIRouter()

# FAISS store
INDEX_PATH = "kb_index.faiss"
DOCS_PATH = "kb_docs.pkl"

if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        doc_texts = pickle.load(f)
else:
    index = faiss.IndexFlatL2(1536)
    doc_texts = []

# Embedding
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Upload Knowledge Base as plain text
@router.post("/chatbot/upload-kb")
async def upload_text_kb(text: str = Form(...)):
    try:
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        for chunk in chunks:
            vector = get_embedding(chunk)
            index.add(np.array([vector], dtype="float32"))
            doc_texts.append(chunk)
        faiss.write_index(index, INDEX_PATH)
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(doc_texts, f)
        return {"message": f"Uploaded {len(chunks)} chunks from text"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add manual problem
@router.post("/chatbot/add-problem")
def add_manual_problem(problem: str = Form(...)):
    db = SessionLocal()
    item = ManualProblem(text=problem)
    db.add(item)
    db.commit()
    db.close()
    return {"message": "Problem added to database"}

# Query model
class ChatQuery(BaseModel):
    question: str

# Query chatbot
@router.post("/chatbot/query")
def query_chatbot(data: ChatQuery):
    db = SessionLocal()
    query = data.question.strip()

    if not query:
        return {"answer": "I'm your assistant. How may I help you today?"}

    # === JD intent detection with flexibility ===
    jd_keywords = ["generate jd", "create jd", "generate a job description"]
    if any(k in query.lower() for k in jd_keywords):
        prompt = f"""
        You are a professional HR assistant.
        Generate a Job Description (JD) based on the following request:

        {query}

        Provide a concise and well-structured JD.
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            answer = response.choices[0].message.content.strip()

            # Save to DB
            db.add(ChatHistory(question=query, response=answer, timestamp=datetime.now()))
            db.commit()
            db.close()
            return {"answer": answer}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # === Otherwise, fallback to knowledge base lookup ===
    try:
        query_embedding = get_embedding(query)
        D, I = index.search(np.array([query_embedding], dtype="float32"), k=3)
        retrieved = [doc_texts[i] for i in I[0] if i < len(doc_texts)]

        # Manual problems
        manual_matches = [p.text for p in db.query(ManualProblem).all() if p.text.lower() in query.lower()]
        context = "\n\n".join(retrieved + manual_matches)

        prompt = f"""
        You are a support chatbot for a product. Answer ONLY from the knowledge below.
        If you don't know the answer or it is unrelated, reply:
        "Sorry, I can't help with that. Please contact support."

        Knowledge Base:
        {context}

        Question:
        {query}
        """

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()

        # Save to DB
        db.add(ChatHistory(question=query, response=answer, timestamp=datetime.now()))
        db.commit()
        db.close()

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.post("/chatbot/voice-query")
async def voice_query(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.mp4', '.ogg')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")

        # Save audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Transcribe
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        query = transcription.text.strip()
        if not query:
            return {"answer": "Sorry, I could not understand the audio."}

        # === JD detection ===
        jd_intent_keywords = ["generate jd", "create jd", "generate a job description", "generate jd for", "generate a jd", "generate a gd"]
        if any(keyword in query.lower() for keyword in jd_intent_keywords):
            prompt = f"""
            You are a professional HR assistant.
            Generate a Job Description (JD) based on the following user request:

            {query}

            Provide a well-structured and concise JD with title, skills, responsibilities, and qualifications.
            """
            jd_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            answer = jd_response.choices[0].message.content.strip()
        else:
            # === Normal chatbot ===
            query_embedding = get_embedding(query)
            D, I = index.search(np.array([query_embedding], dtype="float32"), k=3)
            retrieved = [doc_texts[i] for i in I[0] if i < len(doc_texts)]
            context = "\n\n".join(retrieved)

            prompt = f"""
            You are a support chatbot for a product. Answer ONLY from the knowledge below.
            If you don't know the answer or it is unrelated, reply:
            "Sorry, I can't help with that. Please contact support."

            Knowledge Base:
            {context}

            Question:
            {query}
            """
            chat_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            answer = chat_response.choices[0].message.content.strip()

        # Store chat
        db = SessionLocal()
        db.add(ChatHistory(question=query, response=answer, timestamp=datetime.now()))
        db.commit()
        db.close()

        return {"query": query, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

