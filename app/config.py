"""
config.py – central configuration loaded from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── core ──
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
EMBED_MODEL        = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_DB_DIR      = os.getenv("CHROMA_DB_DIR", "./vector_store")
SERVICE_NAME       = os.getenv("SERVICE_NAME", "MedBuddy")

# ── Firebase ──
FIREBASE_CREDENTIALS_PATH = os.getenv(
    "FIREBASE_CREDENTIALS_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "firebase_credentials.json"),
)

# ── optional services ──
REDIS_URL          = os.getenv("REDIS_URL", "")        # e.g. redis://localhost:6379
LOCAL_MODEL        = os.getenv("LOCAL_MODEL", "")       # HF model id for local GPU LLM
FORCE_CPU          = os.getenv("FORCE_CPU", "0")        # set 1 to force CPU mode
USE_LOCAL_LLM      = os.getenv("USE_LOCAL_LLM", "0")    # set 1 to prefer local model