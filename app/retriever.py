"""
retriever.py – Hybrid retriever (Chroma + BM25) with multi-query expansion,
               re-ranking, symptom-aware Merck retrieval, and auto-tuned
               profile parameters (GPU/CPU).

Includes secondary retrieval passes for:
  • Ayurvedic book treatment chunks
  • Merck Manual (general + targeted symptom/diagnosis/treatment sections)
"""
from app.embedding import embed_text
from app.chroma_client import query_vector
from app.bm25_index import bm25_index
from app.reranker import rerank_candidates
from app.llm_client import llm_generate
from app.profiles import PROFILE
from app.cache import cache_get, cache_set


# ── symptom / query-type detection ────────────────────────────────────
_SYMPTOM_WORDS = {
    "pain", "ache", "fever", "cough", "nausea", "vomiting", "headache",
    "burning", "itching", "swelling", "bleeding", "diarrhea", "constipation",
    "dizzy", "dizziness", "tired", "fatigue", "weakness", "sore", "rash",
    "throat", "stomach", "chest", "back", "joint", "cramp", "numbness",
    "tingling", "breathless", "palpitation", "bloating", "gas", "sneeze",
    "congestion", "chills", "weight loss", "weight gain", "loss of appetite",
    "blurred vision", "anxiety", "insomnia", "heartburn", "acid",
    "inflammation", "stiffness", "bruise", "fracture", "spasm",
    "difficulty swallowing", "shortness of breath", "runny nose",
    "abdominal", "muscle", "bone", "skin", "eye", "ear",
}
_SYMPTOM_PHRASES = [
    "i feel", "i have", "i'm having", "i am having",
    "i've been feeling", "i've been having", "i noticed",
    "it started", "woke up with", "suffering from",
    "hurts", "hurt", "uncomfortable", "discomfort",
    "my head", "my stomach", "my chest", "my back",
    "keeps happening", "getting worse", "won't go away",
]


def detect_query_type(question: str) -> str:
    """Classify a question as 'symptoms', 'diagnosis', 'treatment', or 'general'."""
    q = question.lower()
    if any(p in q for p in _SYMPTOM_PHRASES) or sum(1 for w in _SYMPTOM_WORDS if w in q) >= 1:
        return "symptoms"
    if any(k in q for k in ("what disease", "what condition", "diagnose", "diagnosis",
                            "what do i have", "what's wrong", "could it be", "what causes")):
        return "diagnosis"
    if any(k in q for k in ("how to treat", "treatment", "medicine", "medication",
                            "cure", "remedy", "what should i take", "how to get rid")):
        return "treatment"
    return "general"


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
    1. Classify the query (symptoms / diagnosis / treatment / general)
    2. Retrieve from Chroma (vector) + BM25 (keyword)
    3. **Bonus passes**: Ayurvedic book + Merck (general & section-targeted)
    4. Combine & normalise scores
    5. Re-rank with cross-encoder (top_k controlled by profile)
    """
    if top_k is None:
        top_k = PROFILE["rerank_top_k"]

    per_query_k = PROFILE["candidate_per_query"]
    query_type = detect_query_type(user_question)

    # check cache
    cached = cache_get("retrieve", user_question)
    if cached is not None:
        return cached

    queries = [user_question]
    candidate_map: dict[str, dict] = {}

    for q in queries:
        emb = embed_text(q)[0]

        # --- 1. vector retrieval (all sources) ---
        vec_res = query_vector(emb, n_results=per_query_k)
        for doc_id, doc_text in zip(vec_res["ids"][0], vec_res["documents"][0]):
            if doc_id not in candidate_map:
                candidate_map[doc_id] = {
                    "text": doc_text, "vec_score": 0.0, "bm25_score": 0.0,
                }
            candidate_map[doc_id]["vec_score"] += 1.0

        # --- 2. targeted Ayurvedic book retrieval ---
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
                candidate_map[doc_id]["vec_score"] += 0.5
        except Exception:
            pass

        # --- 3. targeted Merck Manual retrieval ---
        try:
            merck_res = query_vector(
                emb,
                n_results=min(per_query_k, 15),
                where={"source": "The_Merck_clean.txt"},
            )
            for doc_id, doc_text in zip(merck_res["ids"][0], merck_res["documents"][0]):
                if doc_id not in candidate_map:
                    candidate_map[doc_id] = {
                        "text": doc_text, "vec_score": 0.0, "bm25_score": 0.0,
                    }
                candidate_map[doc_id]["vec_score"] += 0.6
        except Exception:
            pass

        # --- 4. BM25 keyword retrieval ---
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

    try:
        cache_set("retrieve", user_question, result, ttl=600)
    except Exception:
        pass

    return result
