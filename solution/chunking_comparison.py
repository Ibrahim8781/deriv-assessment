# solution/chunking_comparison.py
"""Chunking comparison – evaluates two chunking strategies on the same queries.

Strategies:
1. "paragraph" – same as Phase 1 (paragraph + max‑200‑char chunks).
2. "fixed" – deterministic fixed‑size chunks of 200 characters regardless of paragraphs.

For each strategy we build a simple keyword‑overlap index and compute retrieval
metrics (hit/partial_hit/miss) against the ground‑truth titles in queries.json.
The aggregate summary for each strategy is written to
`artifacts/chunking_comparison.json`.
"""

import json
import os
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
KB_DIR = Path(__file__).resolve().parent.parent / "kb"
QUERIES_PATH = BASE_DIR.parent / "queries.json"  # repo root
OUTPUT_PATH = ARTIFACTS_DIR / "chunking_comparison.json"

# Utility tokeniser used for retrieval scoring
def tokenise(text: str):
    return set(re.findall(r"\b\w+\b", text.lower()))

def load_queries():
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def parse_doc(file_path: Path):
    title = ""
    section = ""
    body_lines = []
    with file_path.open("r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Title:"):
            title = line.split(":", 1)[1].strip()
        elif line.startswith("Section:"):
            section = line.split(":", 1)[1].strip()
        elif line == "":
            i += 1
            break
        i += 1
    body = "\n".join(lines[i:]).strip()
    return title, section, body

def chunk_paragraph(body: str, max_len: int = 200):
    chunks = []
    offset = 0
    paragraphs = [p for p in body.split("\n\n") if p]
    for para in paragraphs:
        words = para.split()
        cur = []
        cur_len = 0
        for w in words:
            add = len(w) + (1 if cur else 0)
            if cur_len + add > max_len:
                txt = " ".join(cur)
                start = offset
                end = offset + len(txt)
                chunks.append((txt, start, end))
                offset = end + 1
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len += add
        if cur:
            txt = " ".join(cur)
            start = offset
            end = offset + len(txt)
            chunks.append((txt, start, end))
            offset = end + 1
    return chunks

def chunk_fixed(body: str, size: int = 200):
    chunks = []
    offset = 0
    while offset < len(body):
        txt = body[offset : offset + size]
        start = offset
        end = offset + len(txt) - 1
        chunks.append((txt, start, end))
        offset += size
    return chunks

def build_chunks(strategy: str):
    all_chunks = []
    for fp in KB_DIR.glob("*.txt"):
        title, section, body = parse_doc(fp)
        if not body:
            continue
        if strategy == "paragraph":
            raw_chunks = chunk_paragraph(body)
        else:  # fixed
            raw_chunks = chunk_fixed(body)
        for idx, (txt, start, end) in enumerate(raw_chunks, start=1):
            chunk_id = f"{fp.stem}_{strategy}_c{idx}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_title": title,
                "section": section,
                "text": txt,
                "start_char": start,
                "end_char": end,
            })
    return all_chunks

def retrieve_top_k(question: str, chunks, k=3):
    q_tokens = tokenise(question)
    scored = []
    for c in chunks:
        score = len(q_tokens & tokenise(c.get("text", "")))
        scored.append({"doc_title": c["doc_title"], "chunk_id": c["chunk_id"], "score": score})
    scored.sort(key=lambda x: (-x["score"], x["chunk_id"]))
    return scored[:k]

def evaluate(strategy_chunks, queries):
    hits = partial_hits = misses = 0
    total = len(queries)
    for q in queries:
        expected = set(q.get("expected_doc_titles", []))
        top = retrieve_top_k(q["question"], strategy_chunks, k=3)
        top_titles = {c["doc_title"] for c in top}
        intersect = expected & top_titles
        if intersect == expected:
            hits += 1
        elif intersect:
            partial_hits += 1
        else:
            misses += 1
    return {
        "strategy": "paragraph" if "paragraph" in strategy_chunks[0]["chunk_id"] else "fixed",
        "top3_hit_rate": hits / total if total else 0,
        "total_queries": total,
        "hits": hits,
        "partial_hits": partial_hits,
        "misses": misses,
    }

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    queries = load_queries()
    paragraph_chunks = build_chunks("paragraph")
    fixed_chunks = build_chunks("fixed")
    results = []
    results.append(evaluate(paragraph_chunks, queries))
    results.append(evaluate(fixed_chunks, queries))
    # also include a brief explanation of trade‑offs
    for r in results:
        r["notes"] = (
            "Paragraph chunks preserve natural breaks, likely better recall but may produce more chunks. "
            "Fixed chunks are simpler and produce uniform size, but can split sentences." if r["strategy"] == "paragraph" else
            "Fixed chunks are uniform and faster to compute, but may cut semantics."
        )
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote chunking comparison to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
