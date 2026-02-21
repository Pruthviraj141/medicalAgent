"""
config.py – central configuration loaded from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "mistralai/mixtral-8x7b-instruct")
EMBED_MODEL        = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_DB_DIR      = os.getenv("CHROMA_DB_DIR", "./vector_store")
MONGO_URI          = os.getenv("MONGO_URI", "mongodb://localhost:27017")
SERVICE_NAME       = os.getenv("SERVICE_NAME", "MedBuddy")