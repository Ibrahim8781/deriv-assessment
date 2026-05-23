# solution/phase2.py
"""Phase 2: Retrieve top‑k chunks for each query.

Retrieval strategy: simple deterministic keyword‑overlap count (case‑insensitive)."""

import json
import os
from pathlib import Path

CHUNKS_PATH = Path(__file__).resolve().parent / "artifacts" / "chunks.json"
QUERIES_PATH = Path(__file__).resolve().parent / "queries.json"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
RETRIEVAL_PATH = ARTIFACTS_DIR / "retrieval.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str):
    # simple whitespace tokenisation, lower‑cased, strip punctuation
    import re
    return re.findall(r"\b\w+\b", text.lower())


def score_chunk(query_tokens, chunk_tokens):
    # numeric score = number of shared tokens
    return len(set(query_tokens) & set(chunk_tokens)
)


def retrieve():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    chunks = load_json(CHUNKS_PATH)
    queries = load_json(QUERIES_PATH)
    # pre‑tokenise chunk texts once
    for c in chunks:
        c["_tokens"] = tokenize(c.get("text", ""))
    results = []
    for q in queries:
        q_tokens = tokenize(q.get("question", ""))
        # compute scores
        scored = []
        for c in chunks:
            sc = score_chunk(q_tokens, c["_tokens"])
            scored.append({
                "rank": None,  # placeholder, will fill after sorting
                "chunk_id": c["chunk_id"],
                "doc_title": c["doc_title"],
                "score": float(sc),
                "chunk_text": c["text"],
            })
        # sort descending by score, then by chunk_id for deterministic tie‑breaker
        scored.sort(key=lambda x: (-x["score"], x["chunk_id"]))
        top_k = []
        for rank, item in enumerate(scored[:3], start=1):
            item["rank"] = rank
            top_k.append(item)
        results.append({
            "query_id": q["query_id"],
            "question": q["question"],
            "top_k": top_k,
        })
    # write output
    with RETRIEVAL_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote retrieval results for {len(results)} queries to {RETRIEVAL_PATH}")


if __name__ == "__main__":
    retrieve()
