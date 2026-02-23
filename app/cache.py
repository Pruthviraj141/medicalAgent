"""
cache.py – optional Redis caching for embeddings, LLM responses, and rerank results.

If Redis is not available the app runs without caching (graceful fallback).
Set REDIS_URL in .env to enable.
"""
import os
import hashlib
import json

_redis_available = False
_redis = None

try:
    import redis as _redis_lib
    _redis_url = os.environ.get("REDIS_URL", "")
    if _redis_url:
        _redis = _redis_lib.from_url(_redis_url, decode_responses=True)
        _redis.ping()
        _redis_available = True
        print("✅ Redis connected – response caching enabled")
except Exception as exc:
    print(f"⚠️  Redis unavailable ({exc}). Caching disabled – no impact on functionality.")


def _make_key(*parts: str) -> str:
    raw = "|".join(parts)
    return "medbuddy:" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def cache_get(namespace: str, key_data: str):
    """Return cached JSON value or None."""
    if not _redis_available:
        return None
    try:
        val = _redis.get(_make_key(namespace, key_data))
        return json.loads(val) if val else None
    except Exception:
        return None


def cache_set(namespace: str, key_data: str, value, ttl: int = 3600):
    """Store a JSON-serialisable value with TTL (seconds)."""
    if not _redis_available:
        return
    try:
        _redis.set(_make_key(namespace, key_data), json.dumps(value), ex=ttl)
    except Exception:
        pass
