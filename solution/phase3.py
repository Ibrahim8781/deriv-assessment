# solution/phase3.py
"""Phase 3: Generate citation‑strict answers using Groq LLM.

For each query we feed the retrieved top‑k chunks to the model and ask for an
answer that:
  * uses only the provided chunks,
  * cites each factual statement with the format [doc_title §chunk_id],
  * returns the controlled answer label (grounded_answer or insufficient_context).

The script writes two artifacts:
  * solution/artifacts/answers.json – the final answer records.
  * solution/llm_calls.jsonl   – one line per LLM call with required metadata.
"""

import os
import json
import hashlib
import datetime
import re
from pathlib import Path
import requests
from dotenv import load_dotenv

# Load API key from .env (located at repository root)
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    raise RuntimeError("GROK_API_KEY not found in .env")

# Paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RETRIEVAL_PATH = ARTIFACTS_DIR / "retrieval.json"
QUERIES_PATH = BASE_DIR / "queries.json"
ANSWERS_PATH = ARTIFACTS_DIR / "answers.json"
LLM_LOG_PATH = BASE_DIR / "llm_calls.jsonl"

# Ensure output directories exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_prompt(question: str, chunks: list) -> str:
    """Create a deterministic prompt that lists the retrieved chunks.
    The model is instructed to answer *only* with information from these chunks.
    """
    lines = [f"Question: {question}\n", "Retrieved chunks (use ONLY these):"]
    for idx, c in enumerate(chunks, start=1):
        # Ensure deterministic ordering – chunks are already ordered by rank
        citation = f"[{c['doc_title']} §{c['chunk_id']}]"
        lines.append(f"{idx}. {citation}: {c['chunk_text']}")
    lines.append(
        "\nAnswer the question using only the above chunks.\n"
        "If the information is insufficient, respond with the label 'insufficient_context' and no answer text.\n"
        "Otherwise, provide a concise answer and end each factual sentence with its citation.\n"
        "Return ONLY the answer text (no extra explanation)."
    )
    return "\n".join(lines)


def call_groq(messages: list) -> str:
    headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    # Extract assistant's message content
    return data["choices"][0]["message"]["content"].strip()


def extract_citations(text: str):
    # Find all occurrences of [doc_title §chunk_id]
    pattern = r"\[([^\]]+ §[^\]]+)\]"
    return re.findall(pattern, text)


def process_query(entry, all_chunks):
    query_id = entry["query_id"]
    question = entry["question"]
    top_k = entry.get("top_k", [])
    # Build prompt using retrieved chunks
    prompt = build_prompt(question, top_k)
    # Prepare messages for Groq (system + user)
    messages = [
        {"role": "system", "content": "You are a citation‑strict assistant. Answer using ONLY the provided chunks. Use the exact citation format [doc_title §chunk_id]. If you cannot answer, output the label 'insufficient_context' without any additional text."},
        {"role": "user", "content": prompt},
    ]
    # Call LLM
    answer_text = call_groq(messages)
    # Determine label
    if "insufficient_context" in answer_text.lower():
        answer_label = "insufficient_context"
        answer = ""
        citations = []
        used_chunk_ids = []
    else:
        answer_label = "grounded_answer"
        answer = answer_text
        citations = extract_citations(answer_text)
        # Derive used chunk ids from citations
        used_chunk_ids = [c.split(' §')[1] for c in citations]
    # Build answer record
    answer_record = {
        "query_id": query_id,
        "answer_label": answer_label,
        "answer": answer,
        "citations": citations,
        "used_chunk_ids": used_chunk_ids,
    }
    # Log LLM call
    log_entry = {
        "stage": "ANSWER_GENERATION",
        "query_id": query_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "provider": "groq",
        "model": MODEL,
        "prompt_hash": hash_prompt(prompt),
        "input_artifacts": [str(RETRIEVAL_PATH), str(QUERIES_PATH)],
        "output_artifact": str(ANSWERS_PATH),
    }
    return answer_record, log_entry


def main():
    retrieval = read_json(RETRIEVAL_PATH)
    # queries file also contains extra metadata, but we only need ids
    queries = read_json(QUERIES_PATH)
    answers = []
    llm_logs = []
    for entry in retrieval:
        ans_rec, log_rec = process_query(entry, retrieval)
        answers.append(ans_rec)
        llm_logs.append(log_rec)
    # Write answers
    with ANSWERS_PATH.open("w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    # Append LLM logs (JSON Lines)
    with LLM_LOG_PATH.open("a", encoding="utf-8") as f:
        for log in llm_logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
    print(f"Wrote {len(answers)} answers to {ANSWERS_PATH}")
    print(f"Logged {len(llm_logs)} LLM calls to {LLM_LOG_PATH}")


if __name__ == "__main__":
    main()
