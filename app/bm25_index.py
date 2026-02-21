"""
bm25_index.py – BM25 keyword index kept in RAM alongside Chroma
"""
from rank_bm25 import BM25Okapi
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import word_tokenize


class BM25Index:
    def __init__(self):
        self.docs: list[str] = []
        self.doc_ids: list[str] = []
        self.tokenized: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    # ---- mutators ----
    def add(self, doc_id: str, text: str):
        self.docs.append(text)
        self.doc_ids.append(doc_id)
        self.tokenized.append(word_tokenize(text.lower()))

    def build(self):
        """(Re-)build the BM25 model from current tokenized corpus."""
        if not self.tokenized:
            self.bm25 = None
            return
        self.bm25 = BM25Okapi(self.tokenized)

    # ---- query ----
    def search(self, query: str, top_k: int = 10):
        if self.bm25 is None:
            return []
        q_toks = word_tokenize(query.lower())
        scores = self.bm25.get_scores(q_toks)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (self.doc_ids[idx], self.docs[idx], float(score))
            for idx, score in ranked
        ]


# module-level singleton
bm25_index = BM25Index()
