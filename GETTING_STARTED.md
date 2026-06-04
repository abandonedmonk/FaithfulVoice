# Getting Started with FaithfulVoice

This guide walks you through setting up and running the project from a fresh clone.

## Prerequisites

- **Python 3.10+**
- **Docker** and **Docker Compose**
- **Git**

## 1. Clone and Set Up the Environment

```bash
git clone <repo-url>
cd FaithfulVoice

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser (used for SEC filing HTML processing)
playwright install chromium
```

## 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys. At minimum, set:

- **`NEO4J_PASSWORD`** — must match `docker-compose.yml` (default: `password`)
- **`LLM_API_KEY`** / **`FIREWORKS_API_KEY`** — if using Fireworks as your LLM provider
- **`GROQ_API_KEY`** — only if using Groq vision fallback

If you are using the default Modal endpoints (already configured in `config.yaml`), you do **not** need an LLM API key.

## 3. Start the Data Stores (Qdrant + Neo4j)

```bash
docker compose up -d qdrant neo4j
```

This starts:
- **Qdrant** on `localhost:6333` — vector database for chunk storage
- **Neo4j** on `localhost:7687` (Bolt) / `localhost:7474` (Browser) — graph database for entity/community report storage

Verify they are running:
```bash
docker compose ps
```

## 4. Load Existing Processed Data

The repo already includes processed chunk files and graph checkpoints, so you do not need to re-download or re-process SEC filings to get started.

### Load chunks into Qdrant

```bash
python -m ingestion.scripts.ingest --recreate
```

### Build Neo4j graph from existing checkpoints

```bash
python -m ingestion.scripts.run_neo4j_pipeline --skip-extraction
```

To preview which chunk files the graph pipeline will read without actually running it:

```bash
python -m ingestion.scripts.run_neo4j_pipeline --dry-run
```

## 5. Verify the Setup

Run a quick retrieval test in Python:

```python
from retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(strict_hybrid=False)
results = retriever.retrieve(
    "What changed in NVIDIA revenue growth?",
    top_k=5,
    metadata_filter={"ticker": "NVDA"},
)

print(f"Retrieved {len(results)} results")
```

If you get results back, everything is working.

## Optional: Fill Missing Intel Chunks

Intel (`INTC`) raw filings exist but their processed chunks are missing. To generate them:

```bash
python -m ingestion.scripts.process_all_filings --ticker INTC --force
```

## Common Commands Reference

| Task | Command |
|------|---------|
| Start databases | `docker compose up -d qdrant neo4j` |
| Stop databases | `docker compose down` |
| Load chunks into Qdrant | `python -m ingestion.scripts.ingest --recreate` |
| Build Neo4j graph | `python -m ingestion.scripts.run_neo4j_pipeline --skip-extraction` |
| Preview graph pipeline | `python -m ingestion.scripts.run_neo4j_pipeline --dry-run` |
| Process Intel chunks | `python -m ingestion.scripts.process_all_filings --ticker INTC --force` |

## Troubleshooting

### Qdrant or Neo4j won't start
- Make sure ports `6333`, `7474`, and `7687` are not already in use.
- Run `docker compose logs qdrant` or `docker compose logs neo4j` for details.

### Connection refused when querying
- Confirm both containers are healthy: `docker compose ps`
- Neo4j takes longer to start — wait for the healthcheck to pass.

### Missing `.env` file
- Run `cp .env.example .env` and edit the values.

## Next Steps

- See [README.md](README.md) for full project structure and retrieval examples.
- See [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md) for detailed module-by-module build instructions.
- See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for experiment configurations.
