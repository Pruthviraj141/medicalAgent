"""
ingest.py – Universal ingestion pipeline for ALL data files.

Handles:
  • ALL .txt / .md files in data/ (including The_Merck_clean.txt)
  • book.json (Ayurvedic structured data via book_parser)

Uses fast paragraph-based chunking + batch embedding.
No more dedicated parsers or skip-lists — every text file gets ingested.
"""
import os
import hashlib
import time

from app.chroma_client import collection
from app.embedding import embed_text
from app.bm25_index import bm25_index
from app.book_parser import parse_book_to_chunks, BOOK_JSON
from app.profiles import PROFILE

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ── Fast paragraph-based chunker ──────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """
    Fast paragraph-aware chunking.
    Splits by double newlines (paragraphs), merges small ones,
    splits oversized paragraphs by single newlines, with character overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # If adding this paragraph stays within chunk_size, merge
        if len(current) + len(para) + 2 < chunk_size:
            current = (current + "\n\n" + para) if current else para
        else:
            # Save current chunk
            if current.strip():
                chunks.append(current.strip())

            # If paragraph itself is too large, split by single newlines
            if len(para) > chunk_size:
                lines = para.split("\n")
                current = ""
                for line in lines:
                    if len(current) + len(line) + 1 < chunk_size:
                        current = (current + "\n" + line) if current else line
                    else:
                        if current.strip():
                            chunks.append(current.strip())
                        current = line
            else:
                # Start new chunk with overlap from previous
                if chunks:
                    prev = chunks[-1]
                    overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                    current = overlap_text + "\n\n" + para
                else:
                    current = para

    # Don't forget the last chunk
    if current.strip():
        chunks.append(current.strip())

    return chunks


# ── File hash for change detection ────────────────────────────────────

def _file_hash(filepath: str) -> str:
    """Quick MD5 hash of file contents."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


# ── Ingest ALL text files (universal) ─────────────────────────────────

def _ingest_text_files() -> int:
    """
    Ingest every .txt and .md file in data/ folder.
    Uses fast paragraph chunking + batch embedding.
    Returns total chunks ingested.
    """
    extensions = (".txt", ".md")
    text_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(extensions)
    )

    if not text_files:
        print("  ⚠️  No .txt/.md files found in data/")
        return 0

    print(f"  📄 Found {len(text_files)} text file(s): {', '.join(text_files)}")

    total_chunks = 0
    batch_size = PROFILE["embed_batch_size"]
    chunk_size = PROFILE.get("chunk_max_chars", 1200)

    for fname in text_files:
        filepath = os.path.join(DATA_DIR, fname)
        file_start = time.time()

        # Read file
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            print(f"  ❌ Error reading {fname}: {e}")
            continue

        if not text.strip():
            print(f"  ⚠️  {fname} is empty, skipping")
            continue

        # Chunk
        chunks = _chunk_text(text, chunk_size=chunk_size, overlap=200)
        if not chunks:
            print(f"  ⚠️  {fname} produced 0 chunks, skipping")
            continue

        file_hash = _file_hash(filepath)

        # Batch embed — fast
        print(f"  📘 {fname}: {len(chunks)} chunks, embedding...", end=" ", flush=True)
        all_embeddings = embed_text(chunks, batch_size=batch_size)

        # Prepare IDs and metadata
        ids = [f"{fname}_{file_hash[:8]}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": fname,
                "chunk_index": i,
                "file_hash": file_hash,
                "char_count": len(c),
            }
            for i, c in enumerate(chunks)
        ]

        # Store in ChromaDB in batches
        BATCH = 500
        for b in range(0, len(chunks), BATCH):
            end = min(b + BATCH, len(chunks))
            collection.add(
                ids=ids[b:end],
                embeddings=all_embeddings[b:end],
                documents=chunks[b:end],
                metadatas=metadatas[b:end],
            )

        # Also add to BM25
        for doc_id, chunk_text in zip(ids, chunks):
            bm25_index.add(doc_id, chunk_text)

        elapsed = time.time() - file_start
        print(f"done ({elapsed:.1f}s)")
        total_chunks += len(chunks)

    return total_chunks


# ── Ingest book.json ──────────────────────────────────────────────────

def _ingest_book_json() -> int:
    """
    Ingest the structured Ayurvedic book JSON.
    Returns chunk count.
    """
    if not os.path.isfile(BOOK_JSON):
        print("  ⚠️  book.json not found – skipping Ayurvedic book ingestion.")
        return 0

    book_chunks = parse_book_to_chunks(BOOK_JSON)
    if not book_chunks:
        return 0

    batch_size = PROFILE["embed_batch_size"]

    # Batch embed all chunks
    texts = [c["text"] for c in book_chunks]
    print(f"  📗 Ayurvedic book: {len(texts)} chunks, embedding...", end=" ", flush=True)
    all_embeddings = embed_text(texts, batch_size=batch_size)

    # Prepare
    ids = [c["id"] for c in book_chunks]
    metas = [c["metadata"] for c in book_chunks]

    # Store in ChromaDB in batches
    BATCH = 500
    for b in range(0, len(texts), BATCH):
        end = min(b + BATCH, len(texts))
        collection.add(
            ids=ids[b:end],
            embeddings=all_embeddings[b:end],
            documents=texts[b:end],
            metadatas=metas[b:end],
        )

    # Also add to BM25
    for doc_id, text in zip(ids, texts):
        bm25_index.add(doc_id, text)

    overviews = sum(1 for c in book_chunks if c["metadata"].get("chunk_type") == "overview")
    treatments = sum(1 for c in book_chunks if c["metadata"].get("chunk_type") == "treatment")
    print(f"done ({overviews} overviews, {treatments} treatments)")
    return len(texts)


# ── Master ingestion ─────────────────────────────────────────────────

def ingest_all():
    """Ingest all text files + book.json from the data/ folder."""
    if not os.path.isdir(DATA_DIR):
        print(f"⚠️  Data directory not found: {DATA_DIR}")
        return

    start = time.time()

    text_chunks = _ingest_text_files()
    book_chunks = _ingest_book_json()

    bm25_index.build()
    total = text_chunks + book_chunks
    elapsed = time.time() - start
    print(f"✅ Ingestion complete — {total} chunks total "
          f"({text_chunks} from text files, {book_chunks} from book.json) "
          f"in {elapsed:.1f}s")


if __name__ == "__main__":
    ingest_all()