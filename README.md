# FaithfulVoice

Real-time faithfulness verification for voice-driven financial RAG systems.

This repo now also includes the necessary PRISM GraphRAG artifacts for SEC filing retrieval:
raw filings, processed chunk files, graph checkpoints, ingestion code, retrieval code, Modal serving code, and eval fixtures. The PRISM import intentionally excludes Kubernetes, Terraform, API routes, LangGraph agents, virtualenvs, caches, and duplicate `data/extra` chunk copies.

## Project Structure

```
FaithfulVoice/
├── src/                    # FaithfulVoice RAG, verifier, and voice pipeline
├── ingestion/              # Imported PRISM SEC ingestion + GraphRAG indexing
│   ├── core/               # cleaning, parsing, chunking, entity extraction, Neo4j graph build
│   ├── scripts/            # download/process/ingest/run_neo4j_pipeline entrypoints
│   └── dataset/            # chunk JSONL schema/conversion helpers
├── retrieval/              # Imported PRISM Qdrant + Neo4j retrievers and reranker
├── modal/                  # Imported PRISM Modal services for LLM, vision, embeddings, rerank
├── eval/                   # Imported PRISM eval set, runner, and saved result
├── data/
│   ├── raw/                # SEC filing HTML snapshots, including cleaned HTML variants
│   ├── processed/          # *_chunks.jsonl files used by Qdrant and GraphRAG
│   ├── graphs/             # entity/community report checkpoints
│   └── queries/            # FaithfulVoice question datasets
├── docker-compose.yml      # Local Qdrant + Neo4j only; no API service
├── config.py               # PRISM-compatible config loader
├── config.yaml             # FaithfulVoice + PRISM runtime config
└── .env.example            # Secret placeholders
```

## Imported Data Status

The imported PRISM snapshot contains:

- 35 original raw SEC filing HTML files across AAPL, AMD, AMZN, GOOGL, INTC, META, MSFT, NVDA, and TSLA.
- Cleaned HTML variants for most filings.
- 32 processed chunk files in `data/processed/`.
- Graph extraction and community report checkpoints in `data/graphs/`.

Known gaps preserved from the source data:

- Intel (`INTC`) raw filings exist, but no Intel chunk files were present: `INTC_2023_10K`, `INTC_2024_10K`, `INTC_Q3_2024_10Q`, `INTC_Q4_2024_10Q`.
- The query datasets include `JPM`, but no JPM raw filings or processed chunks were present in the PRISM snapshot.
- `data/processed/NVDA_2024_10K_clean_chunks.jsonl` exists without a matching original raw stem; it was kept because it came from the source snapshot.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
```

Set `NEO4J_PASSWORD=password` in `.env` if using the included `docker-compose.yml` unchanged.

Start local stores:

```bash
docker compose up -d qdrant neo4j
```

## Using Existing Processed Data

The processed chunks are already copied, so you do not need to re-download or re-chunk filings unless you want to fill gaps.

To load existing chunks into Qdrant:

```bash
python -m ingestion.scripts.ingest --recreate
```

To rebuild Neo4j from the existing entity extraction checkpoint:

```bash
python -m ingestion.scripts.run_neo4j_pipeline --skip-extraction
```

To preview which chunk files the graph pipeline will read:

```bash
python -m ingestion.scripts.run_neo4j_pipeline --dry-run
```

## Filling Missing Intel Chunks

Intel raw files were imported, but their processed chunks were missing in the source snapshot. To generate only those chunk files:

```bash
python -m ingestion.scripts.process_all_filings --ticker INTC --force
```

If that script version does not support ticker filtering, run the processor normally and review changed files before committing:

```bash
python -m ingestion.scripts.process_all_filings --force
```

## Retrieval

Hybrid retrieval uses Qdrant vectors plus Neo4j graph context:

```python
from retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(strict_hybrid=False)
results = retriever.retrieve(
    "What changed in NVIDIA revenue growth?",
    top_k=5,
    metadata_filter={"ticker": "NVDA"},
)
```

Global graph retrieval uses community reports:

```python
from retrieval.graph_retriever import GraphRetriever
from retrieval.hybrid_retriever import HybridRetriever

embedding = HybridRetriever(strict_hybrid=False)._embed("major themes across filings")
graph = GraphRetriever()
context = graph.global_search(query_embedding=embedding, top_k=5)
```

## What Was Intentionally Skipped

The import leaves out PRISM `agents/`, `api/`, `k8s/`, `terraform/`, `.venv/`, `.git/`, `.env`, `__pycache__/`, and duplicate `data/extra/` files. This keeps the repo focused on data, ingestion, retrieval, evaluation, and Modal model serving.
