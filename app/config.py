"""Configuration utilities."""
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

CHROMA_DB_PATH = "./vector_store"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

LLM_MODEL = "mistralai/mixtral-8x7b-instruct"