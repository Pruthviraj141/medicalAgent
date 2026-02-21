"""
embedding.py – sentence-transformer BGE embeddings with L2-normalisation
"""
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL
import numpy as np

_model = SentenceTransformer(EMBED_MODEL)


def embed_text(texts):
    """Return list[list[float]] for one or more input strings."""
    if isinstance(texts, str):
        texts = [texts]
    embs = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    return (embs / norms).tolist()