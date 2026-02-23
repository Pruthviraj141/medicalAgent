"""
reranker.py – GPU/CPU-aware cross-encoder re-ranker.

On GPU: full cross-encoder reranking for best accuracy.
On CPU: fast score-passthrough fallback for speed (avoids slow inference).
"""
from app.device import get_device, has_gpu

_DEVICE = get_device()

if has_gpu():
    from sentence_transformers import CrossEncoder
    _MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _cross = CrossEncoder(_MODEL_NAME, device=_DEVICE)
    print(f"🔁 Reranker loaded on {_DEVICE} (cross-encoder)")

    def rerank_candidates(query: str, candidates: list[tuple]):
        """
        Re-score candidates with a cross-encoder (GPU).

        Parameters
        ----------
        query : str
        candidates : list[(doc_id, text, old_score)]

        Returns
        -------
        list[(doc_id, text, cross_score)]  sorted best-first
        """
        if not candidates:
            return []
        pairs = [(query, c[1]) for c in candidates]
        scores = _cross.predict(pairs, show_progress_bar=False)
        ranked = [
            (doc_id, text, float(sc))
            for (doc_id, text, _), sc in zip(candidates, scores)
        ]
        return sorted(ranked, key=lambda x: x[2], reverse=True)

else:
    print("🔁 Reranker: fast CPU fallback (score passthrough)")

    def rerank_candidates(query: str, candidates: list[tuple]):
        """
        Fast CPU fallback: keep existing hybrid scores, just pass through
        the top candidates without expensive cross-encoder inference.
        """
        if not candidates:
            return []
        # already sorted by hybrid score — return as-is
        return [(doc_id, text, score) for doc_id, text, score in candidates[:5]]
