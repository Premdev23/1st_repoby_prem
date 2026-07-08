import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="legal_cases_v2",
    metadata={"hnsw:space": "cosine"}
)


def add_chunks_batch(chunk_ids: list, embeddings: list, metadatas: list, texts: list):
    """Batch upsert chunks into ChromaDB."""
    collection.upsert(
        ids=chunk_ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts
    )


def search_chunks(query_embedding: list, top_k: int = 50) -> dict:
    """Search for top_k most similar chunks."""
    count = collection.count()
    if count == 0:
        return {"metadatas": [[]], "distances": [[]], "documents": [[]], "ids": [[]]}
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["metadatas", "distances", "documents"]
    )


def get_chunk_count() -> int:
    """Return total number of chunks stored."""
    return collection.count()


def reset_collection():
    """Delete all data — used when re-indexing from scratch."""
    global collection
    client.delete_collection("legal_cases_v2")
    collection = client.get_or_create_collection(
        name="legal_cases_v2",
        metadata={"hnsw:space": "cosine"}
    )
    print("🗑️  ChromaDB collection reset.")
