from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from pydantic import BaseModel
import os, tempfile, pickle, faiss, openai, httpx, asyncio
import numpy as np
from datetime import datetime
from openai import OpenAI
from models import SessionLocal, ChatHistory, ManualProblem
from utils import get_embedding, index, doc_texts, detect_intent_and_route
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
router = APIRouter()

# --------------- Upload KB ---------------
@router.post("/chatbot/upload-kb")
async def upload_text_kb(text: str = Form(...)):
    try:
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        for chunk in chunks:
            vector = get_embedding(chunk)
            index.add(np.array([vector], dtype="float32"))
            doc_texts.append(chunk)
        faiss.write_index(index, "kb_index.faiss")
        with open("kb_docs.pkl", "wb") as f:
            pickle.dump(doc_texts, f)
        return {"message": f"Uploaded {len(chunks)} chunks from text"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------- Manual Problem ---------------
@router.post("/chatbot/add-problem")
def add_manual_problem(problem: str = Form(...)):
    db = SessionLocal()
    db.add(ManualProblem(text=problem))
    db.commit()
    db.close()
    return {"message": "Problem added to database"}

# --------------- Chat Query ---------------
class ChatQuery(BaseModel):
    question: str

@router.post("/chatbot/query")
def query_chatbot(data: ChatQuery):
    query = data.question.strip()
    if not query:
        return {"answer": "I'm your assistant. How may I help you today?"}

    intent = detect_intent_and_route(query)

    try:
        if intent == "generate_jd":
            jd_response = asyncio.run(call_generate_jd("Software Engineer", "Python, FastAPI, REST", "2 years"))
            return {"query": query, "answer": jd_response["job_description"]}

        elif intent == "parse_resume":
            return {"answer": "Please upload a resume file to /parse-resume to continue."}
        elif intent == "find_duplicates":
            return {"answer": "Please provide resume folder path to /find-duplicate-resumes to proceed."}
        elif intent == "match_resumes":
            return {"answer": "Please send resume folder and JD to /match-resumes to get results."}
        elif intent == "job_embed":
            return {"answer": "Send JD title, description, and skills to /job-embed to get embedding."}
        else:
            # Fallback: Knowledge base search
            return search_kb_and_respond(query)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------- Voice Query ---------------
@router.post("/chatbot/voice-query")
async def voice_query(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.mp4', '.ogg', '.webm')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        os.remove(tmp_path)

        query = transcription.text.strip()
        if not query:
            return {"answer": "Sorry, I could not understand the audio."}

        intent = detect_intent_and_route(query)

        if intent == "generate_jd":
            jd_response = await call_generate_jd("AI Engineer", "AWS, Langchain, LLMs, Cloud", "4 years")
            answer = jd_response["job_description"]
        elif intent == "parse_resume":
            answer = "Please upload a resume to `/parse-resume` endpoint to continue."
        elif intent == "find_duplicates":
            answer = "Please provide resume folder path to `/find-duplicate-resumes` to proceed."
        elif intent == "match_resumes":
            answer = "Please send resume folder and JD to `/match-resumes` for results."
        elif intent == "job_embed":
            answer = "Please send JD title, skills and description to `/job-embed` to generate embedding."
        else:
            answer = search_kb_and_respond(query)["answer"]

        db = SessionLocal()
        db.add(ChatHistory(question=query, response=answer, timestamp=datetime.now()))
        db.commit()
        db.close()

        return {"query": query, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------- Helpers ---------------
def search_kb_and_respond(query: str):
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
    return {"answer": chat_response.choices[0].message.content.strip()}

async def call_generate_jd(job_title, skills, experience):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/generate-jd", data={
            "job_title": job_title,
            "skills": skills,
            "experience": experience
        })
        return response.json()
