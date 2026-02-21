import os
import uuid
from app.rag_engine import rag_engine

DATA_PATH = "data"


def read_files():

    documents = []

    for file in os.listdir(DATA_PATH):

        filepath = os.path.join(DATA_PATH, file)

        if file.endswith(".txt"):

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(text)

    return documents


def chunk_text(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def ingest_documents():

    documents = read_files()

    for doc in documents:

        chunks = chunk_text(doc)

        for chunk in chunks:

            doc_id = str(uuid.uuid4())

            rag_engine.add_document(chunk, doc_id)

    print("Documents ingested successfully")


if __name__ == "__main__":
    ingest_documents()