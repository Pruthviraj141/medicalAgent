from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_engine import rag_engine

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")

def ask_question(request: QueryRequest):

    result = rag_engine.query(request.question)

    return {
        "status": "success",
        "medical_response": result["answer"],
        "followups": result["followup_questions"],
        "sources": result["sources"]
    }