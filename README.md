# Deriv Assessment RAG Pipeline

This repository contains a lightweight Retrieval‑Augmented Generation (RAG) pipeline built with **Python** and **FastAPI**.

## Overview
- **Phase 1** – Chunk the knowledge‑base (`kb/*.txt`) into small, deterministic pieces (`artifacts/chunks.json`).
- **Phase 2** – Retrieve the top‑3 most relevant chunks for each query using **TF‑IDF** (scikit‑learn).
- **Phase 3** – Generate citation‑strict answers via the **Groq** `llama‑3.3‑70b‑versatile` model.  All LLM calls are logged in `artifacts/llm_calls.jsonl` with relative paths.
- **Phase 4** – Evaluate retrieval performance and perform a grounding check.
- **`solution/main.py`** – Orchestrates the four phases with a deterministic state‑machine.
- **`solution/validate.py`** – Validates that all artifacts exist and conform to the required schema.

## Quick start
```bash
# Install dependencies (Python 3.12)
pip install -r requirements.txt

# Set your Groq API key in a .env file at the repo root
GROK_API_KEY=your_key_here

# Run the full pipeline
python solution/main.py
```

## API (Stretch goal)
A minimal FastAPI service is provided in `solution/api.py`:
```bash
uvicorn solution.api:app --host 0.0.0.0 --port 8000
```
POST `/answer` with JSON `{ "question": "..." }` to receive a grounded answer and citations.

## Repository layout
```
Deriv-Assesment/
├─ kb/                 # Knowledge‑base documents
├─ solution/           # Pipeline phases, orchestrator, API
│   ├─ artifacts/      # Generated JSON artifacts
│   ├─ phase1.py
│   ├─ phase2.py
│   ├─ phase3.py
│   ├─ phase4.py
│   ├─ api.py
│   ├─ main.py
│   └─ validate.py
├─ .gitignore
├─ .env                # (ignored) Groq API key
├─ queries.json        # Query set (moved to repo root)
└─ README.md
```

## License
MIT – feel free to adapt and extend.
