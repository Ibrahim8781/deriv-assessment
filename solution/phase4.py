# solution/phase4.py
"""Phase 4: Deterministic evaluation of retrieval and grounding check.

- Reads `artifacts/retrieval.json`, `queries.json`, and `artifacts/answers.json`.
- For each query computes retrieval metrics against the ground‑truth titles.
- Writes `artifacts/eval.json` (per‑query records + aggregate summary).
- Performs a simple grounding check: verifies that every citation in an answer
  refers to a chunk that was actually retrieved for that query and that the
  cited chunk text shares at least one non‑stopword token with the answer.
  Results are written to `artifacts/grounding_check.json`.
"""

import json
import os
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RETRIEVAL_PATH = ARTIFACTS_DIR / "retrieval.json"
QUERIES_PATH = BASE_DIR / "queries.json"
ANSWERS_PATH = ARTIFACTS_DIR / "answers.json"
EVAL_PATH = ARTIFACTS_DIR / "eval.json"
GROUNDING_PATH = ARTIFACTS_DIR / "grounding_check.json"

# Simple stop‑word list for overlap heuristic
STOP_WORDS = set(
    "the a an and or but if of in on for with to at from by about as into like after before".split()
)

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def tokenise(text: str):
    return [t.lower() for t in re.findall(r"\b\w+\b", text) if t.lower() not in STOP_WORDS]

def evaluate_retrieval(retrieval, queries):
    per_query = []
    hits = partial_hits = misses = 0
    for q in queries:
        qid = q["query_id"]
        expected = set(q.get("expected_doc_titles", []))
        # find retrieval entry
        ret_entry = next((r for r in retrieval if r["query_id"] == qid), None)
        if not ret_entry:
            continue
        top3_titles = [c["doc_title"] for c in ret_entry.get("top_k", [])][:3]
        top3_set = set(top3_titles)
        intersect = expected & top3_set
        if intersect == expected:
            status = "hit"
            hits += 1
        elif intersect:
            status = "partial_hit"
            partial_hits += 1
        else:
            status = "miss"
            misses += 1
        # explanation – first matching rank if any
        explanation = "Expected title not found"
        if intersect:
            # find first rank where a match occurs
            for item in ret_entry.get("top_k", []):
                if item["doc_title"] in expected:
                    explanation = f"Expected title found at rank {item['rank']}"
                    break
        per_query.append({
            "query_id": qid,
            "expected_doc_titles": list(expected),
            "retrieved_doc_titles_top3": top3_titles,
            "retrieval_status": status,
            "matched_expected_title": bool(intersect),
            "explanation": explanation,
        })
    total = len(queries)
    summary = {
        "top3_hit_rate": hits / total if total else 0,
        "total_queries": total,
        "hits": hits,
        "partial_hits": partial_hits,
        "misses": misses,
    }
    return per_query, summary

def grounding_check(answers, retrieval):
    # map query_id -> set of allowed citations (doc_title §chunk_id)
    citation_map = {}
    for ret in retrieval:
        qid = ret["query_id"]
        allowed = set()
        for c in ret.get("top_k", []):
            citation = f"{c['doc_title']} §{c['chunk_id']}"
            allowed.add(citation)
        citation_map[qid] = allowed
    results = []
    for ans in answers:
        qid = ans["query_id"]
        citations = set(ans.get("citations", []))
        allowed = citation_map.get(qid, set())
        missing = list(citations - allowed)
        # overlap heuristic: at least one token overlap between answer and each cited chunk
        overlap_ok = True
        # retrieve chunk texts for the cited chunks
        cited_texts = []
        for ret in retrieval:
            if ret["query_id"] != qid:
                continue
            for c in ret.get("top_k", []):
                cit = f"{c['doc_title']} §{c['chunk_id']}"
                if cit in citations:
                    cited_texts.append(c["chunk_text"])
        answer_tokens = set(tokenise(ans.get("answer", "")))
        for txt in cited_texts:
            if not answer_tokens.intersection(set(tokenise(txt))):
                overlap_ok = False
                break
        grounded = not missing and overlap_ok
        results.append({
            "query_id": qid,
            "grounded": grounded,
            "missing_citations": missing,
            "overlap_ok": overlap_ok,
        })
    return results

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    retrieval = load_json(RETRIEVAL_PATH)
    queries = load_json(QUERIES_PATH)
    answers = load_json(ANSWERS_PATH)
    # Evaluation
    eval_records, summary = evaluate_retrieval(retrieval, queries)
    eval_output = {"per_query": eval_records, "summary": summary}
    with EVAL_PATH.open("w", encoding="utf-8") as f:
        json.dump(eval_output, f, ensure_ascii=False, indent=2)
    # Grounding check
    grounding = grounding_check(answers, retrieval)
    with GROUNDING_PATH.open("w", encoding="utf-8") as f:
        json.dump(grounding, f, ensure_ascii=False, indent=2)
    print(f"Wrote evaluation to {EVAL_PATH}\nWrote grounding check to {GROUNDING_PATH}")

if __name__ == "__main__":
    main()
