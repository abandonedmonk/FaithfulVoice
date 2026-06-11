"""Modal app for GPU-accelerated embedding of SEC filing chunks.

Usage:
    modal deploy scripts/modal_embedder.py

Then run locally:
    python scripts/ingest_from_modal.py
"""
import modal

# Modal image with embedding dependencies (built once, cached on Modal)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "sentence-transformers",   # GPU dense embedding via PyTorch
    "torch",
    "fastembed==0.8.0",        # Sparse BM25 embedding (CPU is fine)
    "numpy",
)

app = modal.App("faithfulvoice-embedder", image=image)


@app.function(gpu="T4", timeout=3600)
def embed_dense(texts: list[str]) -> list[list[float]]:
    """Encode texts with BAAI/bge-base-en-v1.5 on GPU.

    Returns a list of 768-dim normalized float vectors.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    model.max_seq_length = 512

    embeddings = model.encode(
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


@app.function(timeout=3600)
def embed_sparse(texts: list[str]) -> list[dict]:
    """Encode texts with Qdrant/bm25 sparse embedding on CPU.

    Returns a list of dicts with 'indices' and 'values'.
    """
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding("Qdrant/bm25")
    results = list(model.embed(texts))

    return [
        {"indices": r.indices.tolist(), "values": r.values.tolist()}
        for r in results
    ]


@app.function(gpu="T4", timeout=3600)
def embed_batch(texts: list[str]) -> dict:
    """Hybrid dense + sparse embedding in one remote call."""
    dense = embed_dense.local(texts)
    sparse = embed_sparse.local(texts)
    return {"dense": dense, "sparse": sparse}
