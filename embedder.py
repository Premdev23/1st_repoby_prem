from sentence_transformers import SentenceTransformer

# Legal-aware embedding model (384 dimensions, fast, accurate)
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> list:
    """Embed a single text (used for search queries)."""
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list) -> list:
    """Batch embed multiple texts (used during ingestion — much faster)."""
    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False
    ).tolist()
