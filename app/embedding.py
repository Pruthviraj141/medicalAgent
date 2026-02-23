"""
embedding.py – GPU/CPU-aware sentence-transformer BGE embeddings
with L2-normalisation and batch support.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL
from app.device import get_device

_device = get_device()
_model = SentenceTransformer(EMBED_MODEL, device=_device)
print(f"📐 Embedding model loaded on {_device}")


def embed_text(texts, batch_size: int = 64):
    """
    Embed one or more strings and return normalised vectors.

    Parameters
    ----------
    texts : str | list[str]
    batch_size : int – larger batches are faster on GPU.

    Returns
    -------
    list[list[float]]
    """
    if isinstance(texts, str):
        texts = [texts]
    embs = _model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,   # sentence-transformers does L2 internally
    )
    return embs.tolist()