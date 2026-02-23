"""
chroma_client.py – ChromaDB persistent vector store helpers
"""
import chromadb
from app.config import CHROMA_DB_DIR
from app.embedding import embed_text

client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

collection = client.get_or_create_collection(
    name="medical_knowledge",
    metadata={"hnsw:space": "cosine"},
)


def add_chunk(doc_id: str, text: str, metadata: dict | None = None):
    """Embed and upsert a single text chunk."""
    emb = embed_text(text)[0]
    collection.add(
        documents=[text],
        embeddings=[emb],
        ids=[doc_id],
        metadatas=[metadata or {}],
    )


def query_vector(embedding: list[float], n_results: int = 10, where: dict | None = None):
    """
    Query the collection by a pre-computed embedding vector.

    Parameters
    ----------
    embedding : list[float]
    n_results : int
    where : dict | None
        Optional Chroma metadata filter, e.g. {"source": "book.json"}
        or {"chunk_type": "treatment"}.
    """
    kwargs = {"query_embeddings": [embedding], "n_results": n_results}
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)
