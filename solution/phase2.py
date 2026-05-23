# solution/phase2.py
"""Phase 2: Retrieve top‑k chunks for each query.

Retrieval strategy: simple deterministic keyword‑overlap count (case‑insensitive)."""

import json
import os
from pathlib import Path

CHUNKS_PATH = Path(__file__).resolve().parent / "artifacts" / "chunks.json"
# Queries are now at repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = REPO_ROOT / "queries.json"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
RETRIEVAL_PATH = ARTIFACTS_DIR / "retrieval.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str):
    # simple whitespace tokenisation, lower‑cased, strip punctuation
    import re
    return re.findall(r"\b\w+\b", text.lower())

# The TF‑IDF based scoring will be performed later; keep tokenisation helper.


def retrieve():
    """Retrieve top‑3 most similar chunks for each query using TF‑IDF cosine similarity.
    The function writes the results to `artifacts/retrieval.json` keeping the same schema.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    chunks = load_json(CHUNKS_PATH)
    queries = load_json(QUERIES_PATH)
    # Prepare corpus for TF‑IDF: each chunk's text
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    chunk_texts = [c.get("text", "") for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    chunk_matrix = vectorizer.fit_transform(chunk_texts)

    results = []
    for q in queries:
        query_vec = vectorizer.transform([q.get("question", "")])
        sims = cosine_similarity(query_vec, chunk_matrix).flatten()
        # Pair each similarity with its chunk
        scored = []
        for idx, sim in enumerate(sims):
            c = chunks[idx]
            scored.append({
                "rank": None,
                "chunk_id": c["chunk_id"],
                "doc_title": c["doc_title"],
                "score": float(sim),
                "chunk_text": c["text"],
            })
        # Sort by similarity descending, then deterministic chunk_id tie‑breaker
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
