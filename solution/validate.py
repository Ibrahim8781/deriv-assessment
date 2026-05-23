import json
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
VALID_ANSWER_LABELS = {"grounded_answer", "insufficient_context"}
VALID_RETRIEVAL_STATUSES = {"hit", "partial_hit", "miss"}


def _load(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_artifacts() -> None:
    """Validate all required artifacts. Raises AssertionError on failure."""

    # ── chunks.json ──────────────────────────────────────────────────────────
    chunks_path = ARTIFACTS_DIR / "chunks.json"
    assert chunks_path.is_file(), f"Missing {chunks_path}"
    chunks = _load(chunks_path)
    assert isinstance(chunks, list) and len(chunks) > 0, "chunks.json must be a non-empty list"
    required_chunk_keys = {"chunk_id", "doc_title", "section", "text", "start_char", "end_char"}
    for c in chunks:
        missing = required_chunk_keys - c.keys()
        assert not missing, f"Chunk missing keys: {missing}"

    # ── retrieval.json ────────────────────────────────────────────────────────
    retrieval_path = ARTIFACTS_DIR / "retrieval.json"
    assert retrieval_path.is_file(), f"Missing {retrieval_path}"
    retrieval = _load(retrieval_path)
    assert isinstance(retrieval, list) and len(retrieval) > 0, "retrieval.json must be a non-empty list"

    retrieval_map = {}  # query_id -> set of "doc_title §chunk_id"
    for r in retrieval:
        assert {"query_id", "question", "top_k"} <= r.keys(), f"Retrieval entry missing keys: {r}"
        top_k = r["top_k"]
        assert isinstance(top_k, list) and len(top_k) >= 3, \
            f"Query {r['query_id']} must have at least 3 retrieved chunks, got {len(top_k)}"
        # Check scores are numeric
        for chunk in top_k:
            assert isinstance(chunk.get("score"), (int, float)), \
                f"score must be numeric in chunk {chunk.get('chunk_id')}"
        retrieval_map[r["query_id"]] = {
            f"{c['doc_title']} \u00a7{c['chunk_id']}" for c in top_k
        }

    # ── answers.json ──────────────────────────────────────────────────────────
    answers_path = ARTIFACTS_DIR / "answers.json"
    assert answers_path.is_file(), f"Missing {answers_path}"
    answers = _load(answers_path)
    assert isinstance(answers, list) and len(answers) > 0, "answers.json must be a non-empty list"

    # Count check: every query in retrieval has an answer
    retrieval_qids = {r["query_id"] for r in retrieval}
    answer_qids = {a["query_id"] for a in answers}
    assert retrieval_qids == answer_qids, \
        f"Answers missing for queries: {retrieval_qids - answer_qids}"

    for a in answers:
        assert {"query_id", "answer_label", "answer", "citations", "used_chunk_ids"} <= a.keys(), \
            f"Answer entry missing keys: {a.get('query_id')}"
        label = a["answer_label"]
        assert label in VALID_ANSWER_LABELS, \
            f"Invalid answer_label '{label}' for query {a['query_id']}"

        if label == "grounded_answer":
            # Grounded answers must have at least one citation
            assert len(a["citations"]) >= 1, \
                f"grounded_answer for {a['query_id']} must have at least 1 citation"
            # Citations must only refer to retrieved chunks
            allowed = retrieval_map.get(a["query_id"], set())
            for cit in a["citations"]:
                assert cit in allowed, \
                    f"Citation '{cit}' in {a['query_id']} not found in retrieved chunks"

    # ── eval.json ────────────────────────────────────────────────────────────
    eval_path = ARTIFACTS_DIR / "eval.json"
    assert eval_path.is_file(), f"Missing {eval_path}"
    eval_data = _load(eval_path)
    assert "per_query" in eval_data and "summary" in eval_data, \
        "eval.json must have 'per_query' and 'summary' keys"
    # Aggregate summary must be present
    summary = eval_data["summary"]
    assert {"total_queries", "hits", "partial_hits", "misses", "top3_hit_rate"} <= summary.keys(), \
        "eval.json summary is missing required aggregate keys"
    # Statuses must use controlled vocabulary
    for rec in eval_data["per_query"]:
        status = rec.get("retrieval_status")
        assert status in VALID_RETRIEVAL_STATUSES, \
            f"Invalid retrieval_status '{status}' in eval.json"
    # All queries were processed
    eval_qids = {rec["query_id"] for rec in eval_data["per_query"]}
    assert retrieval_qids == eval_qids, \
        f"Evaluation missing for queries: {retrieval_qids - eval_qids}"

    # ── grounding_check.json ──────────────────────────────────────────────────
    grounding_path = ARTIFACTS_DIR / "grounding_check.json"
    assert grounding_path.is_file(), f"Missing {grounding_path}"
    grounding = _load(grounding_path)
    assert isinstance(grounding, list), "grounding_check.json must be a list"

    print("All artifacts validated successfully.")


if __name__ == "__main__":
    validate_artifacts()
