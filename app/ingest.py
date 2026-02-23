"""
ingest.py – read text files AND structured book JSON from data/,
            sentence-chunk them, batch-embed, and push to Chroma + BM25.
"""
import os
import uuid
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

from app.chroma_client import add_chunk, collection
from app.embedding import embed_text
from app.bm25_index import bm25_index
from app.book_parser import parse_book_to_chunks, BOOK_JSON
from app.profiles import PROFILE

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def chunk_text_to_sentences(
    text: str,
    max_chars: int | None = None,
    overlap_sentences: int | None = None,
):
    """Split text into overlapping sentence-level chunks using profile settings."""
    if max_chars is None:
        max_chars = PROFILE["chunk_max_chars"]
    if overlap_sentences is None:
        overlap_sentences = PROFILE["chunk_overlap_sentences"]

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


def _ingest_text_files() -> int:
    """Ingest every .txt / .md file from the data/ folder. Returns chunk count."""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    total_chunks = 0
    batch_size = PROFILE["embed_batch_size"]

    for fname in files:
        fpath = os.path.join(DATA_DIR, fname)
        text = open(fpath, "r", encoding="utf-8").read()
        chunks = chunk_text_to_sentences(text)

        all_ids = [f"{fname}::{uuid.uuid4()}" for _ in chunks]
        all_embeddings = embed_text(chunks, batch_size=batch_size)

        for doc_id, chunk_text, emb in zip(all_ids, chunks, all_embeddings):
            collection.add(
                documents=[chunk_text],
                embeddings=[emb],
                ids=[doc_id],
                metadatas=[{"source": fname}],
            )
            bm25_index.add(doc_id, chunk_text)
            total_chunks += 1

    return total_chunks


def _ingest_book_json() -> int:
    """
    Ingest the structured Ayurvedic book JSON.

    Each plant is split into:
      • 1 overview chunk  (plant name, vernacular names, ailment list)
      • N treatment chunks (one per ailment, with full treatment text)

    Metadata (plant_name, botanical_name, ailment, chunk_type) is stored in
    Chroma for optional filtered retrieval.
    """
    if not os.path.isfile(BOOK_JSON):
        print("⚠️  book.json not found – skipping Ayurvedic book ingestion.")
        return 0

    book_chunks = parse_book_to_chunks(BOOK_JSON)
    if not book_chunks:
        return 0

    batch_size = PROFILE["embed_batch_size"]

    # batch-embed all chunks for speed
    texts = [c["text"] for c in book_chunks]
    all_embeddings = embed_text(texts, batch_size=batch_size)

    total = 0
    for chunk, emb in zip(book_chunks, all_embeddings):
        collection.add(
            documents=[chunk["text"]],
            embeddings=[emb],
            ids=[chunk["id"]],
            metadatas=[chunk["metadata"]],
        )
        bm25_index.add(chunk["id"], chunk["text"])
        total += 1

    print(f"  📗 Ayurvedic book: {total} chunks indexed "
          f"({sum(1 for c in book_chunks if c['metadata']['chunk_type'] == 'overview')} overviews, "
          f"{sum(1 for c in book_chunks if c['metadata']['chunk_type'] == 'treatment')} treatments).")
    return total


def ingest_all():
    """Ingest text files + book JSON from the data/ folder with batch embedding."""
    if not os.path.isdir(DATA_DIR):
        print(f"⚠️  Data directory not found: {DATA_DIR}")
        return

    text_chunks = _ingest_text_files()
    book_chunks = _ingest_book_json()

    bm25_index.build()
    total = text_chunks + book_chunks
    print(f"✅ Ingestion complete — {total} chunks total "
          f"({text_chunks} from text files, {book_chunks} from book.json).")


if __name__ == "__main__":
    ingest_all()