"""RAG engine logic."""
import chromadb
from app.embedding import embedding_model
from app.llm import llm
from app.followup import generate_followups
from app.config import CHROMA_DB_PATH

class RAGEngine:

    def __init__(self):

        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge"
        )

    def add_document(self, text, doc_id):

        embedding = embedding_model.embed(text)

        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id]
        )

    def query(self, question):

        query_embedding = embedding_model.embed(question)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        retrieved_docs = results["documents"][0]

        context = "\n".join(retrieved_docs)

        answer = llm.generate(f"""
Medical Context:
{context}

Question:
{question}

Provide explainable diagnosis with confidence level.
""")

        followups = generate_followups(question, context)

        return {
            "answer": answer,
            "followup_questions": followups,
            "sources": retrieved_docs
        }

rag_engine = RAGEngine()