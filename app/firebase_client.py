"""
firebase_client.py – Firebase Firestore client for MedBuddy

Handles:
  • User management (create / get)
  • Session tracking (create / list / delete)
  • Conversation history (messages per session)
  • Long-term memory storage (embeddings for semantic recall)

Firestore structure:
  users/{user_id}
    └─ sessions/{session_id}
         └─ messages/{auto_id}
  long_memory/{user_id}
    └─ entries/{auto_id}
"""

import os
import time
import firebase_admin
from firebase_admin import credentials, firestore

from app.config import FIREBASE_CREDENTIALS_PATH

# ── Initialize Firebase ──────────────────────────────────────────────
_cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)

# Safe init: reuse existing app if already initialised (e.g. reload)
try:
    _firebase_app = firebase_admin.get_app()
except ValueError:
    _firebase_app = firebase_admin.initialize_app(_cred)

db = firestore.client()
print("✅ Firebase Firestore connected (project: meddy-database)")


def _sanitize_doc(d: dict) -> dict:
    """Convert Firestore-specific types (timestamps, Sentinels) to JSON-safe values."""
    from google.protobuf.timestamp_pb2 import Timestamp  # noqa: F811
    clean = {}
    for k, v in d.items():
        if v is None:
            clean[k] = None
        elif hasattr(v, 'isoformat'):          # DatetimeWithNanoseconds / datetime
            clean[k] = v.isoformat()
        elif type(v).__name__ == 'Sentinel':    # firestore.SERVER_TIMESTAMP before write
            clean[k] = None
        elif isinstance(v, dict):
            clean[k] = _sanitize_doc(v)
        elif isinstance(v, list):
            clean[k] = [_sanitize_doc(i) if isinstance(i, dict) else i for i in v]
        else:
            clean[k] = v
    return clean


# ──────────────────────────────────────────────────────────────────────
# Users
# ──────────────────────────────────────────────────────────────────────
def create_user(user_id: str, name: str = "") -> dict:
    """Create or update a user document."""
    doc_ref = db.collection("users").document(user_id)
    data = {
        "name": name or user_id,
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    doc_ref.set(data, merge=True)
    # Read back so we get the real timestamp, not the Sentinel
    saved = doc_ref.get().to_dict()
    saved["user_id"] = user_id
    return _sanitize_doc(saved)


def get_user(user_id: str) -> dict | None:
    """Return user dict or None if not found."""
    doc = db.collection("users").document(user_id).get()
    if doc.exists:
        d = doc.to_dict()
        d["user_id"] = doc.id
        return _sanitize_doc(d)
    return None


def ensure_user(user_id: str):
    """Create user document if it doesn't exist yet."""
    if not get_user(user_id):
        create_user(user_id)


def list_all_users() -> list[dict]:
    """Return all users from Firestore."""
    docs = db.collection("users").stream()
    result = []
    for doc in docs:
        d = doc.to_dict()
        d["user_id"] = doc.id
        result.append(_sanitize_doc(d))
    return result


def find_user_by_name(name: str) -> dict | None:
    """Find a user by name (case-insensitive match)."""
    users = list_all_users()
    name_lower = name.strip().lower()
    for u in users:
        if u.get("name", "").strip().lower() == name_lower:
            return u
    return None


# ──────────────────────────────────────────────────────────────────────
# Sessions
# ──────────────────────────────────────────────────────────────────────
def create_session(user_id: str, session_id: str) -> dict:
    """Create a new session under a user."""
    ensure_user(user_id)
    doc_ref = (
        db.collection("users").document(user_id)
        .collection("sessions").document(session_id)
    )
    data = {
        "created_at": firestore.SERVER_TIMESTAMP,
        "last_active": firestore.SERVER_TIMESTAMP,
    }
    doc_ref.set(data, merge=True)
    saved = doc_ref.get().to_dict()
    saved["session_id"] = session_id
    return _sanitize_doc(saved)


def get_sessions(user_id: str) -> list[dict]:
    """List all sessions for a user, most-recent first."""
    sessions_ref = (
        db.collection("users").document(user_id)
        .collection("sessions")
    )
    docs = sessions_ref.order_by(
        "last_active", direction=firestore.Query.DESCENDING
    ).stream()
    result = []
    for doc in docs:
        d = doc.to_dict()
        d["session_id"] = doc.id
        result.append(_sanitize_doc(d))
    return result


def delete_session(user_id: str, session_id: str):
    """Delete a session and all its messages."""
    msgs_ref = (
        db.collection("users").document(user_id)
        .collection("sessions").document(session_id)
        .collection("messages")
    )
    # delete messages first (Firestore doesn't cascade)
    for msg in msgs_ref.stream():
        msg.reference.delete()
    # delete session document
    (
        db.collection("users").document(user_id)
        .collection("sessions").document(session_id)
        .delete()
    )


def ensure_session(user_id: str, session_id: str):
    """Create session if it doesn't exist yet."""
    doc = (
        db.collection("users").document(user_id)
        .collection("sessions").document(session_id)
        .get()
    )
    if not doc.exists:
        create_session(user_id, session_id)


# ──────────────────────────────────────────────────────────────────────
# Messages (conversation history)
# ──────────────────────────────────────────────────────────────────────
def add_message(user_id: str, session_id: str, role: str, text: str):
    """Store a message in Firestore and update session last_active."""
    ensure_session(user_id, session_id)
    msgs_ref = (
        db.collection("users").document(user_id)
        .collection("sessions").document(session_id)
        .collection("messages")
    )
    msgs_ref.add({
        "role": role,
        "text": text,
        "ts": time.time(),
    })
    # touch session last_active
    (
        db.collection("users").document(user_id)
        .collection("sessions").document(session_id)
        .update({"last_active": firestore.SERVER_TIMESTAMP})
    )


def get_messages(user_id: str, session_id: str, limit: int = 50) -> list[dict]:
    """Get conversation messages for a session, ordered by timestamp."""
    msgs_ref = (
        db.collection("users").document(user_id)
        .collection("sessions").document(session_id)
        .collection("messages")
    )
    docs = msgs_ref.order_by("ts").limit(limit).stream()
    return [doc.to_dict() for doc in docs]


# ──────────────────────────────────────────────────────────────────────
# Long-term memory (semantic recall)
# ──────────────────────────────────────────────────────────────────────
def store_long_memory(user_id: str, role: str, text: str, embedding: list[float]):
    """Store a memory entry with its embedding vector for semantic recall."""
    (
        db.collection("long_memory").document(user_id)
        .collection("entries")
        .add({
            "role": role,
            "text": text,
            "embedding": embedding,
            "ts": time.time(),
        })
    )


def get_long_memory_entries(user_id: str) -> list[dict]:
    """Retrieve all long-term memory entries for a user."""
    entries_ref = (
        db.collection("long_memory").document(user_id)
        .collection("entries")
    )
    return [doc.to_dict() for doc in entries_ref.stream()]


def clear_long_memory(user_id: str):
    """Delete all long-term memory for a user."""
    entries_ref = (
        db.collection("long_memory").document(user_id)
        .collection("entries")
    )
    for doc in entries_ref.stream():
        doc.reference.delete()
