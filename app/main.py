"""
main.py – FastAPI REST endpoints for MedBuddy (async + GPU/CPU optimised)
"""
from fastapi import FastAPI
from pydantic import BaseModel
from app.ingest import ingest_all
from app.answerer import compose_answer_async, compose_answer
from app.memory import get_short_memory, compress_memory
from app.device import gpu_info

app = FastAPI(
    title="MedBuddy – Medical RAG Brain",
    description=(
        "Hackathon-ready medical RAG agent with hybrid retrieval, "
        "cross-encoder reranking, short+long memory, GPU/CPU auto-detection, "
        "and friendly doctor persona."
    ),
    version="2.0.0",
)


# ──── request schemas ────
class AskRequest(BaseModel):
    session_id: str
    user_id: str
    question: str


class CompressRequest(BaseModel):
    session_id: str
    user_id: str


# ──── endpoints ────
@app.get("/")
def root():
    return {"service": "MedBuddy", "status": "running 🚀", "device": gpu_info()}


@app.post("/ingest")
def run_ingest():
    """Read all files in data/ and push chunks to Chroma + BM25."""
    ingest_all()
    return {"status": "ok", "detail": "Data ingested into vector store & BM25 index."}


@app.post("/ask")
async def ask(req: AskRequest):
    """Main Q&A endpoint – async, returns friendly medical answer + follow-ups + sources."""
    response_text, candidates = await compose_answer_async(
        req.session_id, req.user_id, req.question
    )
    return {
        "status": "success",
        "response_text": response_text,
        "candidates_count": len(candidates),
    }


@app.get("/short_memory/{session_id}")
def get_short(session_id: str):
    """Return the short-term conversation buffer for a session."""
    return {"short_memory": get_short_memory(session_id)}


@app.post("/compress_memory")
def compress(req: CompressRequest):
    """Compress short-term memory into a long-term clinical summary."""
    summary = compress_memory(req.session_id, req.user_id)
    return {"status": "ok", "summary": summary}


@app.get("/health")
def health():
    """Health check with device info."""
    return {"status": "healthy", "device": gpu_info()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)