import json
import os
from pathlib import Path

# Repository root (one level up from this file)
REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

def _load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def validate_artifacts() -> None:
    """Validate that all required artifacts exist and have the expected structure.
    Raises AssertionError if any check fails.
    """
    # 1. chunks.json
    chunks_path = ARTIFACTS_DIR / "chunks.json"
    assert chunks_path.is_file(), f"Missing {chunks_path}"
    chunks = _load_json(chunks_path)
    assert isinstance(chunks, list) and all(isinstance(c, dict) for c in chunks), "chunks.json must be a list of dicts"
    # required keys per chunk
    required_chunk_keys = {"chunk_id", "doc_title", "section", "text", "start_char", "end_char"}
    for c in chunks:
        missing = required_chunk_keys - c.keys()
        assert not missing, f"Chunk {c.get('chunk_id')} missing keys: {missing}"

    # 2. retrieval.json
    retrieval_path = ARTIFACTS_DIR / "retrieval.json"
    assert retrieval_path.is_file(), f"Missing {retrieval_path}"
    retrieval = _load_json(retrieval_path)
    assert isinstance(retrieval, list), "retrieval.json must be a list"
    for r in retrieval:
        assert {"query_id", "question", "top_k"} <= r.keys(), f"Retrieval entry incomplete: {r}"
        assert isinstance(r["top_k"], list) and len(r["top_k"]) >= 3, "Each query must have at least 3 retrieved chunks"

    # 3. answers.json
    answers_path = ARTIFACTS_DIR / "answers.json"
    assert answers_path.is_file(), f"Missing {answers_path}"
    answers = _load_json(answers_path)
    assert isinstance(answers, list), "answers.json must be a list"
    for a in answers:
        assert {"query_id", "answer_label", "answer", "citations", "used_chunk_ids"} <= a.keys(), f"Answer entry incomplete: {a}"
        assert a["answer_label"] in {"grounded_answer", "insufficient_context"}, "Invalid answer_label"

    # 4. eval.json and grounding_check.json (optional but ensure they exist after phase4)
    for fname in ["eval.json", "grounding_check.json"]:
        p = ARTIFACTS_DIR / fname
        assert p.is_file(), f"Missing {p}"
        # load just to ensure valid JSON
        _ = _load_json(p)

    print("All artifacts validated successfully.")

if __name__ == "__main__":
    validate_artifacts()
