# 🧠 Resume Evaluation System (FastAPI + NLP + Hugging Face Integration)

This project implements a smart job-candidate matching system using **semantic embeddings**, **resume parsing**, and **natural language processing (NLP)**. It provides RESTful APIs to:

- Generate semantic embeddings from job descriptions.
- Evaluate CVs based on semantic similarity and relevance.
- Avoid reprocessing of previously analyzed resumes.
- Reuse embeddings for duplicate job descriptions.
- Generate professional job descriptions via Hugging Face API.
- Extract key details from resumes: First Name, Middle Name, Last Name, Email, Mobile Number, Experience.
- Finds the Duplicate Resumes using Hash-functions.

---

## 🚀 Features

- 📄 Supports PDF and DOCX resume parsing.
- 🧠 Semantic similarity scoring using Sentence Transformers.
- 📊 Relevance scoring with experience weighting.
- ✍️ Job description generation via Hugging Face models (e.g., Mixtral-8x7B).
- 🔁 Embedding caching for duplicate jobs.
- ⚡ FastAPI-powered RESTful backend.
- 🔍 Extracts structured data from resumes (name, email, phone, experience).

---

## 📆 Requirements

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
dotenv
requests
google-generativeai
transformers
google-generativeai
spacy
nameparser
openai>=1.0.0
python-multipart
```

---

## ♻️ Setup Instructions

### 1. 🔃 Clone the Repo

```bash
git clone https://github.com/zorhrm/resumes_ai.git
cd resume_ai
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

### 3. 📆 Install Python Dependencies

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

#### 📅 Request (form-data):

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

#### 📅 Request (form-data):

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

### 3. `/generate-jd` (POST)

**Description:** Generates a professional job description using Hugging Face model.

#### 📅 Request (form-data):

| Field      | Type   | Description            |
| ---------- | ------ | ---------------------- |
| job_title  | string | Job title              |
| skills     | string | Comma-separated skills |
| experience | string | Years of experience    |

#### 📤 Response:

```json
{
  "job_description": "- Software Engineer\n- Company Description: ...\n- Responsibilities:\n- ...\n- Requirements:\n- ...\n- Benefits:\n- ..."
}
```

---

### 4. `/extract-resume-data` (POST)

**Description:** Extracts structured information from a resume file.

#### 📅 Request (form-data):

| Field | Type | Description                 |
| ----- | ---- | --------------------------- |
| file  | file | Resume file (.pdf or .docx) |

#### 📤 Response:

```json
{
  "first_name": "John",
  "middle_name": "Alan",
  "last_name": "Doe",
  "email": "john.doe@example.com",
  "mobile": "+91-9123456780",
  "experience": 5.2
}
```

````
### 5. '/find-duplicate-resumes' (POST) 🆕

Detects duplicate resumes from a folder, even if phone or email has changed.

Request (form-data)

|Field	             | Type	  | Description
|resume_folder_path  | string	| Folder path with resumes

#### 📤 Response:

```json
{
  "duplicates": [
    ["resume1.pdf", "resume1_updated.pdf"],
    ["resume2.docx", "resume2 (1).docx"]
  ]
}
````

---

## 🚀 Step 1: Get Your Gemini API Key

- Go to: https://makersuite.google.com/app
- Sign in with your Google account.
- Click on your profile > "Get API key" or go directly to https://makersuite.google.com/app/apikey.
- Copy the generated API key.
- Store it securely. For example, in your .env file:

```bash
GEMINI_API_KEY=your_api_key_here
```

## 📁 Folder Structure

```bash
├── demo.py                  # FastAPI application
├── requirements.txt
├── processed_cvs.json       # Auto-generated processed CV cache
├── job_embeds.json          # Auto-generated job embedding cache
├── resumes/                 # (Place your test resumes here)
├── .env                     # Hugging Face API key
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
- Only certain Hugging Face models support `text-generation` APIs.
- Use models like `gemini-1.5-flash-latest` for JD generation.

---

## 📬 Contact

Made with ❤️ by Poojit Jagadeesh Nagaloti

> Feel free to contribute, suggest improvements, or open issues!
