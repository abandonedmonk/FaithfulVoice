"""Local orchestrator: reads chunk JSONL files, sends batches to Modal GPU,
receives vectors, and upserts into local Qdrant.

Prerequisites:
    1. Deploy Modal app:   modal deploy scripts/modal_embedder.py
    2. Install locally:    pip install modal qdrant-client pyyaml
    3. Qdrant must be running locally (docker-compose up -d)

Usage:
    python scripts/ingest_from_modal.py [--batch-size 256] [--recreate]
"""
import argparse
import json
import time
import uuid
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector, Distance, VectorParams, SparseVectorParams, Modifier

# Modal client for calling remote functions
import modal

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_client(config: dict | None = None) -> QdrantClient:
    if config is None:
        config = load_config()
    return QdrantClient(config["qdrant"]["host"], port=config["qdrant"]["port"])


def create_collection(client: QdrantClient, config: dict | None = None) -> None:
    if config is None:
        config = load_config()
    col = config["qdrant"]["collection"]
    dim = config["qdrant"]["dense_dim"]

    existing = [c.name for c in client.get_collections().collections]
    if col in existing:
        client.delete_collection(col)

    client.create_collection(
        collection_name=col,
        vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    print(f"Created collection '{col}' with dim={dim}")


def lookup_modal_function():
    """Look up the deployed Modal hybrid embedding function."""
    return modal.Function.from_name("faithfulvoice-embedder", "embed_batch")


def ingest_all(
    processed_dir: str | Path = "data/processed",
    config_path: str | Path = CONFIG_PATH,
    recreate: bool = False,
    batch_size: int = 256,
) -> int:
    config = load_config(config_path)
    client = get_client(config)

    if recreate:
        create_collection(client, config)

    processed_dir = Path(processed_dir)
    jsonl_files = sorted(processed_dir.glob("*_chunks.jsonl"))
    if not jsonl_files:
        print("No *_chunks.jsonl files found")
        return 0

    embed_fn = lookup_modal_function()
    col = config["qdrant"]["collection"]
    total_points = 0

    # Check existing points to resume if needed
    collection_info = client.get_collection(col)
    existing_count = collection_info.points_count
    if existing_count > 0 and not recreate:
        print(f"Collection already has {existing_count} points. Will upsert (overwrite duplicates).\n")

    print(f"Ingesting {len(jsonl_files)} filing(s) into '{col}' via Modal GPU...\n")

    for jsonl_file in jsonl_files:
        chunks = []
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        if not chunks:
            print(f"  Skipping empty file: {jsonl_file.name}")
            continue

        print(f"Processing {jsonl_file.name}: {len(chunks)} chunks")
        file_points = 0

        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            batch = chunks[start:end]
            texts = [c["text"] for c in batch]

            print(f"  Embedding batch {start+1}-{end}/{len(chunks)} ...")
            t0 = time.time()

            # Call Modal GPU function
            result = embed_fn.remote(texts)
            dense_vecs = result["dense"]
            sparse_vecs = result["sparse"]

            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s ({len(batch)/elapsed:.1f} chunks/sec)")

            # Build Qdrant points
            points = []
            for i, chunk in enumerate(batch):
                payload = {k: v for k, v in chunk.items() if k not in ("chunk_id",)}
                for key, val in payload.items():
                    if isinstance(val, Path):
                        payload[key] = str(val)

                points.append(
                    PointStruct(
                        id=chunk.get("chunk_id") or str(uuid.uuid4()),
                        vector={
                            "dense": dense_vecs[i],
                            "sparse": SparseVector(
                                indices=sparse_vecs[i]["indices"],
                                values=sparse_vecs[i]["values"],
                            ),
                        },
                        payload=payload,
                    )
                )

            # Upsert to local Qdrant
            client.upsert(collection_name=col, points=points)
            file_points += len(points)
            print(f"    Upserted {file_points}/{len(chunks)} so far")

        total_points += file_points
        print(f"  Completed {jsonl_file.name}: {file_points} points\n")

    print(f"Done. Total {total_points} points in '{col}'.")
    return total_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest chunks via Modal GPU embedding")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collection")
    parser.add_argument("--batch-size", type=int, default=256, help="Chunks per Modal call")
    args = parser.parse_args()

    ingest_all(
        processed_dir=args.processed_dir,
        config_path=args.config,
        recreate=args.recreate,
        batch_size=args.batch_size,
    )
