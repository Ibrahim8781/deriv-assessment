# solution/phase1.py
"""Phase 1: Load KB documents, parse metadata, chunk deterministically, and write chunks.json.

Chunking strategy: split the body into paragraphs and then into max‑200‑character chunks (preserves natural breaks)."""

import json
import os
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "kb"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.json"


def parse_document(file_path: Path):
    """Extract title, section, and body text from a KB file.
    Expected format:
        Title: <title>\n
        Section: <section>\n
        <blank line>\n
        body ...
    """
    title = ""
    section = ""
    body_lines = []
    with file_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    # simple state machine
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Title:"):
            title = line.split(":", 1)[1].strip()
        elif line.startswith("Section:"):
            section = line.split(":", 1)[1].strip()
        elif line == "":
            # first blank line after metadata signals start of body
            i += 1
            break
        i += 1
    # remaining lines constitute body
    body_lines = lines[i:]
    body = "\n".join(body_lines).strip()
    return title, section, body


def chunk_text(body: str, max_len: int = 200):
    """Deterministically chunk *body*.
    First split on double newlines (paragraphs), then split each paragraph into
    pieces <= max_len characters without breaking words.
    Returns list of (chunk_text, start_char, end_char).
    """
    chunks = []
    offset = 0  # character offset from start of body
    paragraphs = [p for p in body.split("\n\n") if p]
    for para in paragraphs:
        words = para.split()
        current = []
        current_len = 0
        for w in words:
            # +1 for space if not first word
            add_len = len(w) + (1 if current else 0)
            if current_len + add_len > max_len:
                chunk_text = " ".join(current)
                start = offset
                end = offset + len(chunk_text)
                chunks.append((chunk_text, start, end))
                offset = end + 1  # account for a newline that was removed
                current = [w]
                current_len = len(w)
            else:
                current.append(w)
                current_len += add_len
        if current:
            chunk_text = " ".join(current)
            start = offset
            end = offset + len(chunk_text)
            chunks.append((chunk_text, start, end))
            offset = end + 1
    return chunks


def build_chunks():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    all_chunks = []
    for file_path in KB_DIR.glob("*.txt"):
        title, section, body = parse_document(file_path)
        if not body:
            continue
        for idx, (txt, start, end) in enumerate(chunk_text(body)):
            chunk_id = f"{file_path.stem}_c{idx+1}"
            chunk = {
                "chunk_id": chunk_id,
                "doc_title": title,
                "section": section,
                "text": txt,
                "start_char": start,
                "end_char": end,
            }
            all_chunks.append(chunk)
    # write JSON array
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(all_chunks)} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    build_chunks()
