"""
memory.py – short-term (RAM) + long-term (Firebase Firestore) conversation memory
             with automatic memory compression for efficiency.

Replaces MongoDB with Firebase.  Short-term stays in RAM for speed;
every message is also persisted to Firestore so conversation history
survives server restarts.
"""

import time
import numpy as np
from app.embedding import embed_text

# ──────────────────────────────────────────────
# Firebase connection (graceful fallback)
# ──────────────────────────────────────────────
_firebase_available = False

try:
    from app.firebase_client import (
        add_message as _fb_add_message,
        store_long_memory as _fb_store,
        get_long_memory_entries as _fb_get_entries,
        ensure_session as _fb_ensure_session,
    )
    _firebase_available = True
except Exception as exc:
    print(f"⚠️  Firebase unavailable ({exc}). Long-term memory will be skipped.")

# ──────────────────────────────────────────────
# Short-term memory (per session, in RAM)
# ──────────────────────────────────────────────
SESSION_BUFFERS: dict[str, list[dict]] = {}
_MAX_SHORT_MESSAGES = 12


def add_to_short_memory(session_id: str, role: str, text: str,
                        user_id: str = ""):
    """
    Append to the in-RAM conversation buffer **and** persist the message
    to Firebase (if *user_id* is provided and Firebase is available).
    """
    SESSION_BUFFERS.setdefault(session_id, []).append(
        {"role": role, "text": text, "ts": time.time()}
    )
    SESSION_BUFFERS[session_id] = SESSION_BUFFERS[session_id][-_MAX_SHORT_MESSAGES:]

    # also persist to Firestore conversation history
    if _firebase_available and user_id:
        try:
            _fb_add_message(user_id, session_id, role, text)
        except Exception:
            pass  # non-critical — RAM copy still works


def get_short_memory(session_id: str) -> list[dict]:
    return SESSION_BUFFERS.get(session_id, [])


def clear_short_memory(session_id: str):
    SESSION_BUFFERS.pop(session_id, None)


# ──────────────────────────────────────────────
# Long-term memory (Firebase Firestore)
# ──────────────────────────────────────────────
def add_to_long_memory(user_id: str, role: str, text: str):
    if not _firebase_available:
        return
    emb = embed_text(text)[0]
    try:
        _fb_store(user_id, role, text, emb)
    except Exception:
        pass


def recall_long_memory(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    if not _firebase_available:
        return []
    q_emb = np.array(embed_text(query)[0])
    try:
        docs = _fb_get_entries(user_id)
    except Exception:
        return []
    if not docs:
        return []
    sims = [(d, float(np.dot(q_emb, np.array(d["embedding"])))) for d in docs]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in sims[:top_k]]


# ──────────────────────────────────────────────
# Memory compression (summarise long histories)
# ──────────────────────────────────────────────
def compress_memory(session_id: str, user_id: str) -> str | None:
    """
    Summarise the current short-term buffer into a 2-sentence clinical note,
    store it as long-term memory, and clear the short-term buffer.
    Returns the summary text, or None if nothing to compress.
    """
    buf = get_short_memory(session_id)
    if not buf:
        return None

    from app.llm_client import llm_generate

    messages_text = "\n".join(f"{m['role']}: {m['text']}" for m in buf)
    prompt = (
        "Summarise the following patient conversation into a concise 2-sentence "
        "clinical summary. Include key symptoms, conditions discussed, and any "
        "recommendations given:\n\n" + messages_text
    )
    summary = llm_generate(prompt, temperature=0.0, max_tokens=100)

    # store compressed summary as long-term memory
    add_to_long_memory(user_id, "summary", summary)

    # clear short-term buffer
    clear_short_memory(session_id)

    return summary
