from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import uvicorn
import fitz  
import docx
from typing import Optional, List
import os
import hashlib
import json
import torch
import csv
from datetime import datetime
import re

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

PROCESSED_CVS_FILE = "processed_cvs.json"
JOB_EMBEDDINGS_CSV = "job_embeddings.csv"

# Load already processed resumes from JSON
if os.path.exists(PROCESSED_CVS_FILE):
    with open(PROCESSED_CVS_FILE, "r") as f:
        processed_resumes = json.load(f)
else:
    processed_resumes = {}

# Synonym mapping
ROLE_SYNONYMS = {
    "software developer": "software engineer",
    "web developer": "frontend engineer",
    "data scientist": "machine learning engineer"
}

def normalize_role(role: str) -> str:
    role_lower = role.lower().strip()
    return ROLE_SYNONYMS.get(role_lower, role_lower)

def extract_years(text: str) -> float:
    text = text.lower()
    numbers = list(map(float, re.findall(r"\d+(?:\.\d+)?", text)))
    if "between" in text and len(numbers) >= 2:
        return sum(numbers[:2]) / 2
    elif numbers:
        return numbers[0]
    return 0.0

def extract_text_from_pdf(file_path: str) -> str:
    with fitz.open(file_path) as doc:
        return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

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
        "embedding": json.dumps(embedding, indent=2)  # Pretty print
    }

    file_exists = os.path.exists(JOB_EMBEDDINGS_CSV)
    with open(JOB_EMBEDDINGS_CSV, mode="a", newline='', encoding='utf-8') as csvfile:
        fieldnames = ["timestamp", "job_title", "experience", "skills", "description", "embedding"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    return {"embedding": embedding}

@app.post("/match-resumes")
def match_resumes(
    job_description: str = Form(...),
    resume_folder_path: str = Form(...),
    years_experience: str = Form("0")
):
    parsed_experience = extract_years(years_experience)
    job_emb_tensor = model.encode(job_description, convert_to_tensor=True)
    results = []

    for filename in os.listdir(resume_folder_path):
        filepath = os.path.join(resume_folder_path, filename)
        if not os.path.isfile(filepath):
            continue

        file_hash = get_file_hash(filepath)
        if file_hash in processed_resumes:
            continue

        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(filepath)
        else:
            continue

        resume_emb = model.encode(text, convert_to_tensor=True)
        semantic_score = util.pytorch_cos_sim(resume_emb, job_emb_tensor).item()
        experience_bonus = min(parsed_experience / 10, 1.0)
        relevance_score = (semantic_score * 0.8 + experience_bonus * 0.2) * 100

        score = {
            "filename": filename,
            "filepath": filepath,
            "semantic_score": round(semantic_score, 4),
            "experience_bonus": round(experience_bonus, 2),
            "relevance_score": round(relevance_score, 2)
        }

        results.append(score)
        processed_resumes[file_hash] = score

    with open(PROCESSED_CVS_FILE, "w") as f:
        json.dump(processed_resumes, f, indent=4)

    return results

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
