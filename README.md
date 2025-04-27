# 🧠 Resume Evaluation System (FastAPI + NLP)

This project implements a smart job-candidate matching system using **semantic embeddings**, **resume parsing**, and **natural language processing (NLP)**. It provides RESTful APIs to:

- Generate semantic embeddings from job descriptions.
- Evaluate CVs based on semantic similarity and relevance.
- Avoid reprocessing of previously analyzed resumes.
- Reuse embeddings for duplicate job descriptions.

---

## 🚀 Features

- 📄 Supports PDF and DOCX resume parsing.
- 🧠 Semantic similarity scoring using Sentence Transformers.
- 📊 Relevance scoring with experience weighting.
- 🔁 Embedding caching for duplicate jobs.
- ⚡ FastAPI-powered RESTful backend.

---

## 📦 Requirements

- Python 3.8+
- `virtualenv` or `venv` for isolated environment

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
```

### 📄 `requirements.txt`

```text
fastapi
uvicorn
sentence-transformers
python-docx
PyMuPDF
torch
```

---

## 🔁 Setup Instructions

### 1. 🔃 Clone the Repo

```bash
git clone https://github.com/your-username/smart-resume-matcher.git
cd smart-resume-matcher
```

### 2. 🐍 Create Virtual Environment

#### Using `venv`:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

#### Using `virtualenv` (optional):

```bash
virtualenv venv
source venv/bin/activate
```

### 3. 📦 Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Running the Server

```bash
uvicorn main:app --reload
```

- Server will run on `http://127.0.0.1:8000`
- Swagger docs available at `http://127.0.0.1:8000/docs`

---

## 📡 API Endpoints

### 1. `/job-embed` (POST)

**Description:** Generates semantic embedding for a job.

#### 📥 Request (form-data):

| Field       | Type   | Description                   |
| ----------- | ------ | ----------------------------- |
| job_title   | string | Job title                     |
| experience  | string | Experience requirement (text) |
| skills      | string | Required skills               |
| description | string | Full job description          |

#### 📤 Response:

```json
{
  "embedding": [...],
  "job_hash": "hashed_id"
}
```

---

### 2. `/match-resumes` (POST)

**Description:** Matches resumes against job description or job embedding.

#### 📥 Request (form-data):

| Field              | Type   | Description                                      |
| ------------------ | ------ | ------------------------------------------------ |
| job_description    | string | Full job description                             |
| resume_folder_path | string | Path to local folder containing CVs              |
| years_experience   | float  | Applicant's years of experience                  |
| job_embedding      | string | (Optional) JSON-formatted list from `/job-embed` |

#### 📤 Response:

```json
{
  "results": [
    {
      "filename": "John_Doe.pdf",
      "filepath": "resumes/John_Doe.pdf",
      "semantic_score": 0.83,
      "experience_bonus": 0.3,
      "relevance_score": 71.0
    }
  ],
  "total_resumes": 5
}
```

---

## 📁 Folder Structure

```bash
├── main.py                  # FastAPI application
├── requirements.txt
├── processed_cvs.json       # Auto-generated processed CV cache
├── job_embeds.json          # Auto-generated job embedding cache
├── resumes/                 # (Place your test resumes here)
└── README.md
```

---

## 🧪 Testing Locally

1. Add some sample `.pdf` or `.docx` files to a `resumes/` folder.
2. Use Postman or Swagger (`/docs`) to test endpoints.

---

## 🔒 Notes

- Resume data is hashed to avoid duplicate processing.
- Job embeddings are cached via hash (MD5 of full job text).
- Ensure resume files are readable and not encrypted.

---

## 📬 Contact

Made with ❤️ by Poojit Jagadeesh Nagaloti

> Feel free to contribute, suggest improvements, or open issues!
