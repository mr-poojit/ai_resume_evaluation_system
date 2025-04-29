from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from fastapi.responses import JSONResponse
import requests
import json
from dotenv import load_dotenv
import uvicorn
import fitz 
import docx
import os
import hashlib
import json
import torch
import csv
import re
import requests
from datetime import datetime
from dateutil import parser

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

PROCESSED_CVS_FILE = "processed_cvs.json"

# Load processed resumes if available
if os.path.exists(PROCESSED_CVS_FILE):
    with open(PROCESSED_CVS_FILE, "r") as f:
        processed_resumes = json.load(f)
else:
    processed_resumes = {}

# -------------------
# NEW: Huggingface API settings
# -------------------
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # Set this as an environment variable
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
    "Content-Type": "application/json"
}

ROLE_SYNONYMS = {
    "software developer": "software engineer",
    "web developer": "frontend engineer",
    "data scientist": "machine learning engineer"
}

MONTHS_MAPPING = {
    'jan': '01', 'january': '01',
    'feb': '02', 'february': '02',
    'mar': '03', 'march': '03',
    'apr': '04', 'april': '04',
    'may': '05',
    'jun': '06', 'june': '06',
    'jul': '07', 'july': '07',
    'aug': '08', 'august': '08',
    'sep': '09', 'september': '09',
    'oct': '10', 'october': '10',
    'nov': '11', 'november': '11',
    'dec': '12', 'december': '12',
}

def normalize_role(role: str) -> str:
    role_lower = role.lower().strip()
    return ROLE_SYNONYMS.get(role_lower, role_lower)

def extract_years(text: str) -> float:
    text = text.lower()
    exp_phrases = re.findall(r'(\d{1,2}(?:\.\d+)?)\s*\+?\s*(?:years|yrs)', text)
    if exp_phrases:
        numbers = list(map(float, exp_phrases))
        if numbers:
            return max(numbers)
    return 0.0

def extract_experience_from_dates(text: str) -> float:
    text = text.lower()
    now = datetime.now()
    matches = re.findall(r'([a-z]{3,9})\s*(\d{4})\s*(?:-|to|–)\s*([a-z]{3,9}|present|current)\s*(\d{0,4})?', text)

    total_months = 0

    for match in matches:
        start_month_text, start_year_text, end_month_text, end_year_text = match

        if not start_year_text.isdigit():
            continue

        start_month = MONTHS_MAPPING.get(start_month_text[:3], '01')
        start_year = start_year_text

        try:
            start_date = parser.parse(f"{start_year}-{start_month}-01")
        except:
            continue

        if end_month_text in ['present', 'current']:
            end_date = now
        else:
            end_month = MONTHS_MAPPING.get(end_month_text[:3], '01')
            end_year = end_year_text if end_year_text else start_year
            try:
                end_date = parser.parse(f"{end_year}-{end_month}-01")
            except:
                end_date = now

        months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        if 0 < months < 600:
            total_months += months

    total_years = round(total_months / 12, 2)
    return total_years

def calculate_experience_bonus(required_experience: float, actual_experience: float) -> float:
    if actual_experience >= required_experience:
        return 1.0
    elif required_experience == 0:
        return 0.0
    else:
        return round(actual_experience / required_experience, 2)

def extract_text_from_pdf(file_path: str) -> str:
    with fitz.open(file_path) as doc:
        return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def get_combined_hash(file_path: str, job_description: str) -> str:
    with open(file_path, "rb") as f:
        file_content = f.read()
    combined = file_content + job_description.encode('utf-8')
    return hashlib.md5(combined).hexdigest()

# -------------------------------
# NEW: API to generate JD
# -------------------------------
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # Set this as an environment variable
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
    "Content-Type": "application/json"
}

@app.post("/generate-jd")
def generate_job_description(job_title: str = Form(...), skills: str = Form(...), experience: str = Form(...)):
    prompt = (
        f"Generate a professional job description for a {job_title} role.\n"
        f"Required experience: {experience} years.\n"
        f"Skills: {skills}.\n"
        f"Format the output in bullet points under responsibilities and requirements.\n"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 400,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"HuggingFace API Error: {response.status_code} {response.text}")

    generated_text = response.json()[0]["generated_text"]

    # Return escaped `\n` using json.dumps()
    return JSONResponse(content={"job_description": json.dumps(generated_text)})

# -------------------------------
# Existing: API to embed job
# -------------------------------
@app.post("/job-embed")
def get_job_embedding(
    job_title: str = Form(...),
    experience: str = Form(...),
    skills: str = Form(...),
    description: str = Form(...)
):
    norm_job_title = normalize_role(job_title)
    parsed_exp = extract_years(experience)
    full_text = f"{norm_job_title} {parsed_exp} years experience {skills} {description}"
    embedding = model.encode(full_text).tolist()

    row = {
        "timestamp": datetime.now().isoformat(),
        "job_title": job_title,
        "experience": experience,
        "skills": skills,
        "description": description,
        "embedding": json.dumps(embedding, indent=2)
    }

    file_exists = os.path.exists("job_embeddings.csv")
    with open("job_embeddings.csv", mode="a", newline='', encoding='utf-8') as csvfile:
        fieldnames = ["timestamp", "job_title", "experience", "skills", "description", "embedding"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return {"embedding": embedding}

# -------------------------------
# Existing: API to match resumes
# -------------------------------
@app.post("/match-resumes")
def match_resumes(
    job_description: str = Form(...),
    resume_folder_path: str = Form(...),
    years_experience: str = Form("0")
):
    parsed_job_experience = extract_years(years_experience)

    if not os.path.exists(resume_folder_path):
        raise HTTPException(status_code=400, detail="Resume folder path does not exist.")

    job_emb_tensor = model.encode(job_description, convert_to_tensor=True)
    results = []

    for filename in os.listdir(resume_folder_path):
        filepath = os.path.join(resume_folder_path, filename)
        if not os.path.isfile(filepath):
            continue

        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        elif filename.lower().endswith(".docx"):
            text = extract_text_from_docx(filepath)
        else:
            continue

        combined_hash = get_combined_hash(filepath, job_description)

        resume_emb = model.encode(text, convert_to_tensor=True)

        semantic_score = util.pytorch_cos_sim(resume_emb, job_emb_tensor).item()

        actual_experience_years = extract_experience_from_dates(text)
        if actual_experience_years == 0:
            actual_experience_years = extract_years(text)

        experience_bonus = calculate_experience_bonus(parsed_job_experience, actual_experience_years)

        relevance_score = (semantic_score * 0.8 + experience_bonus * 0.2) * 100

        score = {
            "filename": filename,
            "semantic_score": round(semantic_score, 4),
            "actual_experience_years": round(actual_experience_years, 2),
            "experience_bonus": round(experience_bonus, 2),
            "relevance_score": round(relevance_score, 2)
        }

        results.append(score)
        processed_resumes[combined_hash] = score

    with open(PROCESSED_CVS_FILE, "w") as f:
        json.dump(processed_resumes, f, indent=4)

    return results

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
