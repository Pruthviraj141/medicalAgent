"""
retriever.py – Hybrid retriever (Chroma + BM25) with multi-query expansion & re-ranking
"""
from app.embedding import embed_text
from app.chroma_client import query_vector
from app.bm25_index import bm25_index
from app.reranker import rerank_candidates
from app.llm_client import llm_generate


def generate_multi_queries(user_question: str, n: int = 3) -> list[str]:
    """Ask the LLM to paraphrase the user question into *n* short search queries."""
    prompt = (
        f"Paraphrase the following user query into {n} short, distinct search queries "
        f"for retrieving medical text. Return them comma-separated, nothing else.\n"
        f"Query: {user_question}\nOutput:"
    )
    resp = llm_generate(prompt, temperature=0.0, max_tokens=120)
    parts = [p.strip() for p in resp.replace("\n", ",").split(",") if p.strip()]
    return parts[:n] if parts else [user_question]


def hybrid_retrieve(user_question: str, top_k: int = 10):
    """
    1. Generate multi-queries via LLM
    2. Retrieve from Chroma (vector) + BM25 (keyword)
    3. Combine & normalise scores
    4. Re-rank with cross-encoder
    """
    queries = generate_multi_queries(user_question, n=3)

    candidate_map: dict[str, dict] = {}

    for q in queries:
        # --- vector retrieval ---
        emb = embed_text(q)[0]
        vec_res = query_vector(emb, n_results=top_k)
        for doc_id, doc_text in zip(vec_res["ids"][0], vec_res["documents"][0]):
            if doc_id not in candidate_map:
                candidate_map[doc_id] = {
                    "text": doc_text, "vec_score": 0.0, "bm25_score": 0.0,
                }
            candidate_map[doc_id]["vec_score"] += 1.0

        # --- BM25 retrieval ---
        for doc_id, text, score in bm25_index.search(q, top_k):
            if doc_id not in candidate_map:
                candidate_map[doc_id] = {
                    "text": text, "vec_score": 0.0, "bm25_score": 0.0,
                }
            candidate_map[doc_id]["bm25_score"] += score

    # --- normalise & combine (0.6 vector + 0.4 BM25) ---
    max_vec  = max((v["vec_score"]  for v in candidate_map.values()), default=1) or 1
    max_bm25 = max((v["bm25_score"] for v in candidate_map.values()), default=1) or 1

    items = []
    for doc_id, v in candidate_map.items():
        combined = 0.6 * (v["vec_score"] / max_vec) + 0.4 * (v["bm25_score"] / max_bm25)
        items.append((doc_id, v["text"], combined))

    items_sorted = sorted(items, key=lambda x: x[2], reverse=True)[:top_k]

    # --- cross-encoder re-rank ---
    return rerank_candidates(user_question, items_sorted)
