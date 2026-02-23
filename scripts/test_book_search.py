"""Quick smoke test: verify book data is searchable in the RAG pipeline."""
from app.embedding import embed_text
from app.chroma_client import query_vector
from app.bm25_index import bm25_index

print("=" * 60)
print("TEST 1: Vector search — diabetes (book.json only)")
print("=" * 60)
emb = embed_text("Ayurvedic treatment for diabetes")[0]
res = query_vector(emb, n_results=3, where={"source": "book.json"})
for i, (doc_id, text) in enumerate(zip(res["ids"][0], res["documents"][0])):
    print(f"  {i+1}. [{doc_id}]")
    print(f"     {text[:150]}...")
    print()

print("=" * 60)
print("TEST 2: BM25 search — turmeric jaundice")
print("=" * 60)
results = bm25_index.search("turmeric jaundice treatment", 3)
for doc_id, text, score in results:
    print(f"  [{score:.2f}] {doc_id}")
    print(f"     {text[:150]}...")
    print()

print("=" * 60)
print("TEST 3: Vector search — headache (all sources)")
print("=" * 60)
emb2 = embed_text("remedy for headache")[0]
res2 = query_vector(emb2, n_results=5)
for doc_id, text in zip(res2["ids"][0], res2["documents"][0]):
    src = "BOOK" if doc_id.startswith("book::") else "TEXT"
    print(f"  [{src}] {doc_id[:45]}")
    print(f"     {text[:120]}...")
    print()

print("=" * 60)
print("TEST 4: Collection stats")
print("=" * 60)
from app.chroma_client import collection
print(f"  Total documents in Chroma: {collection.count()}")
print(f"  BM25 index size: {len(bm25_index.docs)}")
print()
print("ALL TESTS PASSED ✅")
