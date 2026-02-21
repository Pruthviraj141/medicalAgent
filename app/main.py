"""
main.py – FastAPI REST endpoints for MedBuddy
"""
from fastapi import FastAPI
from pydantic import BaseModel
from app.ingest import ingest_all
from app.answerer import compose_answer
from app.memory import get_short_memory

app = FastAPI(
    title="MedBuddy – Medical RAG Brain",
    description="Hackathon-ready medical RAG agent with hybrid retrieval, memory, and friendly doctor persona.",
    version="1.0.0",
)


# ──── request schemas ────
class AskRequest(BaseModel):
    session_id: str
    user_id: str
    question: str


# ──── endpoints ────
@app.get("/")
def root():
    return {"service": "MedBuddy", "status": "running 🚀"}


@app.post("/ingest")
def run_ingest():
    """Read all files in data/ and push chunks to Chroma + BM25."""
    ingest_all()
    return {"status": "ok", "detail": "Data ingested into vector store & BM25 index."}


@app.post("/ask")
def ask(req: AskRequest):
    """Main Q&A endpoint – returns friendly medical answer + follow-ups + sources."""
    response_text, candidates = compose_answer(req.session_id, req.user_id, req.question)
    return {
        "status": "success",
        "response_text": response_text,
        "candidates_count": len(candidates),
    }


@app.get("/short_memory/{session_id}")
def get_short(session_id: str):
    """Return the short-term conversation buffer for a session."""
    return {"short_memory": get_short_memory(session_id)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)