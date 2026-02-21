"""
reranker.py – lightweight cross-encoder re-ranker
"""
from sentence_transformers import CrossEncoder

_cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_candidates(query: str, candidates: list[tuple]):
    """
    Re-score candidates with a cross-encoder.

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
    scores = _cross.predict(pairs)
    ranked = [
        (doc_id, text, float(sc))
        for (doc_id, text, _), sc in zip(candidates, scores)
    ]
    return sorted(ranked, key=lambda x: x[2], reverse=True)
