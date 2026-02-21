"""
ingest.py – read text files from data/, sentence-chunk them, push to Chroma + BM25
"""
import os
import uuid
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

from app.chroma_client import add_chunk
from app.bm25_index import bm25_index

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def chunk_text_to_sentences(text: str, max_chars: int = 1500, overlap_sentences: int = 2):
    """Split text into overlapping sentence-level chunks."""
    sentences = sent_tokenize(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for sent in sentences:
        if cur_len + len(sent) > max_chars and cur:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur_len = sum(len(s) for s in cur)
        cur.append(sent)
        cur_len += len(sent)

    if cur:
        chunks.append(" ".join(cur))
    return chunks


def ingest_all():
    """Ingest every .txt / .md file from the data/ folder."""
    if not os.path.isdir(DATA_DIR):
        print(f"⚠️  Data directory not found: {DATA_DIR}")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    total_chunks = 0

    for fname in files:
        fpath = os.path.join(DATA_DIR, fname)
        text = open(fpath, "r", encoding="utf-8").read()
        chunks = chunk_text_to_sentences(text)
        for c in chunks:
            doc_id = f"{fname}::{uuid.uuid4()}"
            add_chunk(doc_id, c, metadata={"source": fname})
            bm25_index.add(doc_id, c)
            total_chunks += 1

    bm25_index.build()
    print(f"✅ Ingestion complete — {total_chunks} chunks from {len(files)} file(s).")


if __name__ == "__main__":
    ingest_all()