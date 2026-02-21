"""Embedding utilities."""
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL

class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)

    def embed(self, text: str):
        return self.model.encode(text).tolist()

embedding_model = EmbeddingModel()