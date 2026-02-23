"""
profiles.py – performance presets that auto-tune based on GPU vs CPU.

Import PROFILE anywhere and use its values to control retrieval sizes,
batch sizes, and context windows.
"""
from app.device import has_gpu

if has_gpu():
    PROFILE = {
        # retrieval
        "candidate_per_query": 30,
        "paraphrase_queries": 1,
        "rerank_top_k": 8,
        "final_context_docs": 5,
        # embedding
        "embed_batch_size": 128,
        # chunking
        "chunk_max_chars": 1200,
        "chunk_overlap_sentences": 3,
        # LLM
        "llm_max_tokens": 400,
        "llm_temperature": 0.40,
    }
    print("⚡ GPU performance profile active")
else:
    PROFILE = {
        # retrieval
        "candidate_per_query": 20,
        "paraphrase_queries": 1,
        "rerank_top_k": 5,
        "final_context_docs": 5,
        # embedding
        "embed_batch_size": 32,
        # chunking
        "chunk_max_chars": 1000,
        "chunk_overlap_sentences": 2,
        # LLM
        "llm_max_tokens": 400,
        "llm_temperature": 0.40,
    }
    print("🐢 CPU performance profile active")
