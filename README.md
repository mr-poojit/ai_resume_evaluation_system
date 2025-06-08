# 🧠 Resume Evaluation + AI Chatbot System (FastAPI + OpenAI + Whisper + FAISS)

This project implements an intelligent platform that combines **resume evaluation**, **AI-powered job description generation**, and an **OpenAI-based support chatbot** with **voice query capabilities**.

---

## 🚀 Features

### 🔍 Resume Evaluation

- 📄 Supports PDF and DOCX parsing
- 🧠 Embedding-based matching with Sentence Transformers
- 🎯 Relevance scoring with experience weighting
- ♻️ Caches job/resume embeddings to avoid recomputation
- 📤 Extracts: Name, Email, Phone, Experience
- 🧬 Detects duplicate resumes (even if email/phone differ)

### 🤖 Chatbot (Text + Voice)

- Integrated chatbot powered by OpenAI (GPT-3.5)
- Knowledge upload supported via text chunks
- ❓ Users can ask questions related to uploaded product documents
- 🗣 Voice-based querying using Whisper ASR
- 🎯 Out-of-scope detection: Replies with fallback message
- 📝 Manual problems can be inserted to FAQ from API
- 🧾 All chatbot history stored in SQLite DB

### 📝 JD Generation

- `/generate-jd` API: Generate professional Job Descriptions from job title, skills, and experience
- 🔄 Integrated with voice assistant (e.g. "generate JD for AI Engineer...")

---

## 📦 Requirements

```bash
Python >= 3.8
```

### 📄 `requirements.txt`

```
fastapi
uvicorn
sentence-transformers
python-docx
PyMuPDF
torch
dotenv
requests
google-generativeai
transformers
spacy
nameparser
openai>=1.0.0
faiss-cpu
pydub
whisper
python-multipart
```

---

## ⚙️ Setup

```bash
git clone https://github.com/zorhrm/resumes_ai.git
cd resume_evaluation_system

python -m venv venv
source venv/bin/activate   # Or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

Create a `.env` file in root:

```
OPENAI_API_KEY=sk-...your_key...
```

---

## 🚀 Run the Server

```bash
uvicorn demo:app --reload
```

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 API Endpoints

### ✅ Resume APIs

- `POST /job-embed` — Create job embedding
- `POST /match-resumes` — Match resumes to a job
- `POST /generate-jd` — Generate JD from inputs
- `POST /extract-resume-data` — Extract name, email, phone, exp
- `POST /find-duplicate-resumes` — Detect duplicates

### 💬 Chatbot APIs

- `POST /chatbot/upload-kb` — Upload plain text as knowledge
- `POST /chatbot/add-problem` — Add a manual problem (FAQ)
- `POST /chatbot/query` — Query the chatbot (text)
- `POST /chatbot/voice-query` — Query chatbot using voice

---

## 🧠 How Chatbot Works

1. Upload knowledge as plain text (split into chunks, vectorized using OpenAI embeddings)
2. Voice query or text query comes in
3. Matches the query vector with FAISS
4. If it matches known problems, it responds based on retrieved context
5. All interactions are stored in SQLite database

---

## 🔁 Resetting Memory (Clear FAISS + Docs)

To clear the uploaded knowledge base:

```bash
rm kb_index.faiss kb_docs.pkl   # On Windows use: del
```

You can now upload new KB via `/chatbot/upload-kb`.

---

## 💬 Voice JD Example

> Upload MP3 file saying: “Generate JD for AI Engineer with 5 years experience in AWS, LangChain, and LLMs”

Will return structured JD generated from OpenAI.

---

## 🗃 DB Tables

- `chat_history` — Stores query and response
- `manual_problems` — Stores manually added FAQs

To inspect:

```bash
sqlite3 chatbot.db
.tables
SELECT * FROM chat_history;
```

---

## 👨‍💻 Author

Made with ❤️ by **Poojit Jagadeesh Nagaloti**
