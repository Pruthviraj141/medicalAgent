"""
retriever.py – Hybrid retriever (Chroma + BM25) with multi-query expansion,
               re-ranking, and auto-tuned profile parameters (GPU/CPU).

Includes a secondary Ayurvedic book retrieval pass to boost
structured treatment chunks when relevant.
"""
from app.embedding import embed_text
from app.chroma_client import query_vector
from app.bm25_index import bm25_index
from app.reranker import rerank_candidates
from app.llm_client import llm_generate
from app.profiles import PROFILE
from app.cache import cache_get, cache_set


def generate_multi_queries(user_question: str, n: int | None = None) -> list[str]:
    """Ask the LLM to paraphrase the user question into *n* short search queries."""
    if n is None:
        n = PROFILE["paraphrase_queries"]
    prompt = (
        f"Paraphrase the following user query into {n} short, distinct search queries "
        f"for retrieving medical text. Return them comma-separated, nothing else.\n"
        f"Query: {user_question}\nOutput:"
    )
    resp = llm_generate(prompt, temperature=0.0, max_tokens=120)
    parts = [p.strip() for p in resp.replace("\n", ",").split(",") if p.strip()]
    return parts[:n] if parts else [user_question]


def hybrid_retrieve(user_question: str, top_k: int | None = None):
    """
    1. Generate multi-queries via LLM
    2. Retrieve from Chroma (vector) + BM25 (keyword)
    3. **Bonus pass**: retrieve from book.json treatment chunks specifically
    4. Combine & normalise scores
    5. Re-rank with cross-encoder (top_k controlled by profile)
    """
    if top_k is None:
        top_k = PROFILE["rerank_top_k"]

    per_query_k = PROFILE["candidate_per_query"]

    # check cache
    cached = cache_get("retrieve", user_question)
    if cached is not None:
        return cached

    # speed optimisation: skip multi-query LLM call, use direct question
    queries = [user_question]

    candidate_map: dict[str, dict] = {}

    for q in queries:
        # --- vector retrieval (all sources) ---
        emb = embed_text(q)[0]
        vec_res = query_vector(emb, n_results=per_query_k)
        for doc_id, doc_text in zip(vec_res["ids"][0], vec_res["documents"][0]):
            if doc_id not in candidate_map:
                candidate_map[doc_id] = {
                    "text": doc_text, "vec_score": 0.0, "bm25_score": 0.0,
                }
            candidate_map[doc_id]["vec_score"] += 1.0

        # --- targeted book treatment retrieval ---
        # fetch treatment chunks from the Ayurvedic book specifically
        # this ensures structured herbal remedies surface even when
        # the general corpus is large
        try:
            book_res = query_vector(
                emb,
                n_results=min(per_query_k, 15),
                where={"source": "book.json"},
            )
            for doc_id, doc_text in zip(book_res["ids"][0], book_res["documents"][0]):
                if doc_id not in candidate_map:
                    candidate_map[doc_id] = {
                        "text": doc_text, "vec_score": 0.0, "bm25_score": 0.0,
                    }
                # give a slight boost (0.5) so book results compete fairly
                candidate_map[doc_id]["vec_score"] += 0.5
        except Exception:
            pass  # gracefully degrade if no book data ingested yet

        # --- BM25 retrieval ---
        for doc_id, text, score in bm25_index.search(q, per_query_k):
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
    result = rerank_candidates(user_question, items_sorted)

    # cache result (TTL 10 min)
    try:
        cache_set("retrieve", user_question, result, ttl=600)
    except Exception:
        pass

    return result
