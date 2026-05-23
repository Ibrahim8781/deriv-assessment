# solution/api.py
"""FastAPI minimal API exposing POST /answer.

- Loads pre‑computed chunks from `artifacts/chunks.json` at startup.
- Retrieves top‑3 chunks for the incoming question using the same deterministic
  keyword‑overlap scoring as Phase 2.
- Calls the Groq LLM (`llama-3.3-70b-versatile`) to generate a citation‑strict
  answer. The prompt mirrors Phase 3 and enforces the controlled vocabularies.
- Returns a JSON response with `answer_label`, `answer` and `citations`.

Only the essential dependencies are imported to keep the service lightweight.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import re
import hashlib
import datetime
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (API key)
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    raise RuntimeError("GROK_API_KEY not found in .env")

app = FastAPI()

# Paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.json"

# Load chunks once at startup
with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    CHUNKS = json.load(f)

# Simple tokeniser used for retrieval
def tokenise(text: str):
    return re.findall(r"\b\w+\b", text.lower())

def retrieve_top_k(question: str, k: int = 3):
    q_tokens = set(tokenise(question))
    scored = []
    for c in CHUNKS:
        c_tokens = set(tokenise(c.get("text", "")))
        score = len(q_tokens & c_tokens)
        scored.append({
            "chunk_id": c["chunk_id"],
            "doc_title": c["doc_title"],
            "chunk_text": c["text"],
            "score": float(score),
        })
    scored.sort(key=lambda x: (-x["score"], x["chunk_id"]))
    top = []
    for rank, item in enumerate(scored[:k], start=1):
        item["rank"] = rank
        top.append(item)
    return top

# Prompt construction – identical to Phase 3
def build_prompt(question: str, chunks):
    lines = [f"Question: {question}\n", "Retrieved chunks (use ONLY these):"]
    for idx, c in enumerate(chunks, start=1):
        citation = f"[{c['doc_title']} §{c['chunk_id']}]"
        lines.append(f"{idx}. {citation}: {c['chunk_text']}")
    lines.append(
        "\nAnswer the question using only the above chunks.\n"
        "If the information is insufficient, respond with the label 'insufficient_context' and no answer text.\n"
        "Otherwise, provide a concise answer and end each factual sentence with its citation.\n"
        "Return ONLY the answer text."
    )
    return "\n".join(lines)

# LLM call (same as Phase 3)
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

def call_groq(messages):
    headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def extract_citations(text: str):
    return re.findall(r"\[([^\]]+ §[^\]]+)\]", text)

class AnswerRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer_label: str
    answer: str | None = None
    citations: list[str] = []

@app.post("/answer", response_model=AnswerResponse)
def answer_endpoint(req: AnswerRequest):
    # 1. Retrieve
    top_chunks = retrieve_top_k(req.question, k=3)
    # 2. Build prompt
    prompt = build_prompt(req.question, top_chunks)
    messages = [
        {"role": "system", "content": "You are a citation‑strict assistant. Answer using ONLY the provided chunks. Use the exact citation format [doc_title §chunk_id]. If you cannot answer, output the label 'insufficient_context' without any additional text."},
        {"role": "user", "content": prompt},
    ]
    # 3. Call LLM
    try:
        answer_text = call_groq(messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    # 4. Determine label and citations
    if "insufficient_context" in answer_text.lower():
        label = "insufficient_context"
        answer = None
        citations = []
    else:
        label = "grounded_answer"
        answer = answer_text
        citations = extract_citations(answer_text)
    return AnswerResponse(answer_label=label, answer=answer, citations=citations)
