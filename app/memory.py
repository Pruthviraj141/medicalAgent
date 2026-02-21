"""
memory.py – short-term (RAM) + long-term (MongoDB) conversation memory
"""
import time
import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from app.config import MONGO_URI
from app.embedding import embed_text

# ──────────────────────────────────────────────
# MongoDB connection (graceful fallback)
# ──────────────────────────────────────────────
_mongo_available = False
mem_col = None

try:
    _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    _client.admin.command("ping")
    _db = _client.medbuddy
    mem_col = _db.user_memory
    _mongo_available = True
    print("✅ MongoDB connected – long-term memory enabled")
except (ConnectionFailure, Exception) as exc:
    print(f"⚠️  MongoDB unavailable ({exc}). Long-term memory will be skipped.")

# ──────────────────────────────────────────────
# Short-term memory (per session, in RAM)
# ──────────────────────────────────────────────
SESSION_BUFFERS: dict[str, list[dict]] = {}
_MAX_SHORT_MESSAGES = 12


def add_to_short_memory(session_id: str, role: str, text: str):
    SESSION_BUFFERS.setdefault(session_id, []).append(
        {"role": role, "text": text, "ts": time.time()}
    )
    SESSION_BUFFERS[session_id] = SESSION_BUFFERS[session_id][-_MAX_SHORT_MESSAGES:]


def get_short_memory(session_id: str) -> list[dict]:
    return SESSION_BUFFERS.get(session_id, [])


# ──────────────────────────────────────────────
# Long-term memory (MongoDB)
# ──────────────────────────────────────────────
def add_to_long_memory(user_id: str, role: str, text: str):
    if not _mongo_available:
        return
    emb = embed_text(text)[0]
    mem_col.insert_one(
        {"user_id": user_id, "role": role, "text": text, "embedding": emb, "ts": time.time()}
    )


def recall_long_memory(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    if not _mongo_available:
        return []
    q_emb = np.array(embed_text(query)[0])
    docs = list(mem_col.find({"user_id": user_id}))
    if not docs:
        return []
    sims = [(d, float(np.dot(q_emb, np.array(d["embedding"])))) for d in docs]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in sims[:top_k]]
