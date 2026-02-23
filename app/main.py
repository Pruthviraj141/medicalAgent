"""
main.py – FastAPI REST endpoints for MedBuddy (async + GPU/CPU optimised)
"""
import os
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.ingest import ingest_all
from app.answerer import compose_answer_async, compose_answer
from app.memory import get_short_memory, compress_memory
from app.device import gpu_info


# ──── lifespan (startup / shutdown) ────
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup: init Firebase + auto-ingest data so RAG works immediately."""
    # Firebase
    try:
        from app.firebase_client import db as _fb_db  # noqa: F401
        print("🔥 Firebase ready")
    except Exception as e:
        print(f"⚠️  Firebase init skipped: {e}")

    # Auto-ingest: ensure vector store + BM25 are populated
    from app.chroma_client import collection
    from app.bm25_index import bm25_index
    doc_count = collection.count()
    if doc_count == 0:
        print("📥 No data in vector store — running auto-ingest...")
        ingest_all()
    else:
        print(f"📦 Vector store has {doc_count} chunks — skipping ingest.")
        # Rebuild BM25 from Chroma so keyword search works after restart
        if bm25_index.bm25 is None:
            print("🔄 Rebuilding BM25 index from vector store...")
            all_docs = collection.get()
            for doc_id, doc_text in zip(all_docs["ids"], all_docs["documents"]):
                bm25_index.add(doc_id, doc_text)
            bm25_index.build()
            print(f"✅ BM25 index rebuilt with {len(all_docs['ids'])} docs")

    yield  # ← app is running
    print("👋 Shutting down MedBuddy")


app = FastAPI(
    title="MedBuddy – Medical RAG Brain",
    description=(
        "Hackathon-ready medical RAG agent with hybrid retrieval, "
        "cross-encoder reranking, short+long memory, GPU/CPU auto-detection, "
        "Firebase Firestore persistence, and friendly doctor persona."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# ──── CORS (allow chat UI) ────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──── serve chat UI ────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    """Serve the chat UI."""
    html_path = os.path.join(_PROJECT_ROOT, "chat.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ──── global exception handler (shows real errors in API responses) ────
@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"❌ Unhandled error: {exc}\n{tb}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ──── request schemas ────
class AskRequest(BaseModel):
    session_id: str
    user_id: str
    question: str


class CompressRequest(BaseModel):
    session_id: str
    user_id: str


class CreateUserRequest(BaseModel):
    user_id: str
    name: str = ""


# ──── core endpoints ────
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


# ──── Firebase user / session / conversation endpoints ────
@app.post("/user")
def create_user_endpoint(req: CreateUserRequest):
    """Create or update a user in Firestore."""
    from app.firebase_client import create_user
    user = create_user(req.user_id, req.name)
    return {"status": "ok", "user": user}


@app.get("/user/{user_id}")
def get_user_endpoint(user_id: str):
    """Get user info from Firestore."""
    from app.firebase_client import get_user
    user = get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "ok", "user": user}


@app.get("/sessions/{user_id}")
def list_sessions(user_id: str):
    """List all sessions for a user, most-recent first."""
    from app.firebase_client import get_sessions
    sessions = get_sessions(user_id)
    return {"status": "ok", "sessions": sessions, "count": len(sessions)}


@app.get("/conversation/{user_id}/{session_id}")
def get_conversation(user_id: str, session_id: str):
    """Get full conversation history for a session."""
    from app.firebase_client import get_messages
    messages = get_messages(user_id, session_id)
    return {"status": "ok", "messages": messages, "count": len(messages)}


@app.delete("/session/{user_id}/{session_id}")
def delete_session_endpoint(user_id: str, session_id: str):
    """Delete a session and all its messages."""
    from app.firebase_client import delete_session
    delete_session(user_id, session_id)
    return {"status": "ok", "detail": f"Session {session_id} deleted"}


@app.get("/users")
def list_all_users_endpoint():
    """List all registered users."""
    from app.firebase_client import list_all_users
    users = list_all_users()
    return {"status": "ok", "users": users, "count": len(users)}


@app.get("/find_user/{name}")
def find_user_by_name_endpoint(name: str):
    """Find a user by name (case-insensitive)."""
    from app.firebase_client import find_user_by_name
    user = find_user_by_name(name)
    if user:
        return {"status": "found", "user": user}
    return {"status": "not_found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)