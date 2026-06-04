# FaithfulVoice Data Pipeline

This document describes the full data flow — from raw SEC HTML filings to the final Qdrant vectors and Neo4j graph.

---

## Overview

```
Raw SEC HTML (.htm)
        │
        ▼
  Clean HTML (_clean.htm)          ← strips iXBRL, hidden elements, boilerplate
        │
        ▼
  Chunk JSONL (*_chunks.jsonl)     ← parent/child hierarchy, tables as markdown
        │
        ├──────────────────────┐
        ▼                      ▼
  Qdrant Collection      Neo4j Graph
  (dense + sparse        (entities, relations,
   vectors)               community reports)
```

---

## Step 1: Download Raw SEC Filings

```bash
python -m ingestion.scripts.download_filings --output-dir data/raw
```

**What it does:**
- Hits the SEC EDGAR API (`data.sec.gov/submissions/`) for 10-K and 10-Q filings
- Downloads for 10 companies: NVDA, AMD, INTC, AAPL, MSFT, GOOGL, META, AMZN, TSLA, JPM
- Rate-limited at 0.15s between requests to respect SEC guidelines
- Saves as `data/raw/{TICKER}_{YEAR}_{FORM}.htm` (e.g. `NVDA_2024_10K.htm`)

**Filing specs:**
| Form | Label | What it covers |
|------|-------|----------------|
| 10-K | 2024_10K | Annual report FY2024 |
| 10-K | 2023_10K | Annual report FY2023 |
| 10-Q | Q4_2024_10Q | Quarterly report Q4 2024 |
| 10-Q | Q3_2024_10Q | Quarterly report Q3 2024 |

**Output:** `data/raw/` — raw `.htm` files from SEC EDGAR

---

## Step 2: Clean HTML

```bash
python -m ingestion.scripts.process_all_filings
```

This script runs the full pipeline: clean → chunk → write JSONL.

**Cleaning** (`ingestion/core/cleaner.py` → `clean_edgar_html()`):
1. **Unwrap iXBRL tags** — removes `<ix:*>` tags but keeps their text content (regulatory markup, not human-readable)
2. **Remove hidden elements** — deletes any tag with `display:none` in its style attribute (footnotes, references)
3. **Remove iXBRL header** — deletes `<ix:header>` block (document metadata)
4. **Write cleaned file** — outputs `{original_stem}_clean.htm`

**Output:** `data/raw/NVDA_2024_10K_clean.htm`, etc.

---

## Step 3: Parse & Chunk

Still part of `process_all_filings`, calls `chunk_filing()` from `ingestion/core/chunker.py`.

**Parsing** (`ingestion/core/parser.py`):
- Parses cleaned HTML into typed elements:
  - `NarrativeText` — body paragraphs
  - `Table` — financial tables
  - `ListItem` — bulleted/numbered lists

**Section grouping** (`_group_by_section()`):
- Groups elements by their SEC section anchor ID
- Maps TOC links and colored span headers to sections:
  | Anchor ID | Section |
  |-----------|---------|
  | `item1` | Business Overview |
  | `item1a` | Risk Factors |
  | `item1c` | Cybersecurity |
  | `item2` | Properties |
  | `item3` | Legal Proceedings |
  | `item7` | MD&A |
  | `item7a` | Market Risk |
  | `item8` | Financial Statements |
  | `item9a` | Controls and Procedures |
- Skips boilerplate sections: `cover`, `item4`, `item6`, `item9`, `item9b`, `item9c`, `item10`, `item11`, `item14`, `item16`

**Chunking strategy** — two-level parent/child hierarchy:
| Level | Chunk size | Overlap | Purpose |
|-------|-----------|---------|---------|
| Parent | 3500 tokens | 300 tokens | Broad context for retrieval |
| Child | 1024 tokens | 256 tokens | Dense vector embedding, precise matching |

Splitter uses `langchain_text_splitters.RecursiveCharacterTextSplitter` with separators: `["\n\n", "\n", ". ", " ", ""]`

**Table handling:**
- Converts HTML tables to markdown via `pandas.read_html()` → `.to_markdown()`
- Renders table as PNG image (via `table_renderer.py`)
- Generates vision description of the table image (via `vision_extractor.py`)

**Metadata attached to every chunk:**
```python
Chunk(
    text="child chunk text (1024 tokens)",
    parent_text="parent chunk text (3500 tokens)",
    table_markdown="markdown table if element is Table",
    table_image_path="path to rendered PNG",
    vision_description="LLM-generated table description",
    company="NVIDIA Corporation",
    ticker="NVDA",
    cik="0001045810",
    year=2024,
    quarter=None,
    filing_type="10-K",
    accession_number="0001045810-24-000042",
    section="MD&A",
    anchor_id="item7",
    element_type="NarrativeText",
    chunk_index=0,
    parent_chunk_id="uuid",
    chunk_id="uuid",
    source_url="https://www.sec.gov/...",
    htm_filename="NVDA_2024_10K.htm",
)
```

**Output:** `data/processed/{TICKER}_{YEAR}_{FORM}_chunks.jsonl` — one line per Chunk object, ~800-1000 chunks per 10-K filing

---

## Step 4: Embed & Load into Qdrant

```bash
python -m ingestion.scripts.ingest --recreate
```

**What it does:**
1. Reads all `*_chunks.jsonl` files from `data/processed/`
2. Embeds each chunk's text:
   - **Dense vector**: `BAAI/bge-base-en-v1.5` → 384 dimensions, cosine distance
   - **Sparse vector**: `Qdrant/bm25` → with IDF modifier
   - Embedding can use local FastEmbed or remote Modal endpoint (configured in `config.yaml`)
3. Creates Qdrant collection `prism_filings`:
   ```python
   vectors_config={
       "dense": VectorParams(size=384, distance=Distance.COSINE),
   },
   sparse_vectors_config={
       "sparse": SparseVectorParams(modifier=Modifier.IDF),
   },
   ```
4. Builds `PointStruct` for each chunk:
   - `id` = chunk's UUID
   - `vector` = `{"dense": [...], "sparse": {"indices": [...], "values": [...]}}`
   - `payload` = full chunk metadata dict (ticker, section, anchor_id, parent_text, table_markdown, etc.)
5. Upserts points in batches of 200

**Embedding text for tables:**
- If chunk is a Table with a vision description: `f"{vision_description}\n\n{table_markdown}"`
- Otherwise: just the chunk `text` field

**Output:** Qdrant collection `prism_filings` persisted in Docker volume `prism_qdrant_data` (112MB on disk)

**Verification:**
```bash
curl http://localhost:6333/collections/prism_filings
```

---

## Step 5: Build Neo4j Graph

```bash
python -m ingestion.scripts.run_neo4j_pipeline --skip-extraction
```

Four sub-steps:

### 5a. Entity Extraction (skipped with `--skip-extraction`)
- Uses LLM (`Qwen/Qwen2.5-14B-Instruct`) to extract entities and relations from chunks
- Runs in batches (default: 8 chunks per batch, 8 concurrent)
- Saves checkpoint: `data/graphs/graph_extractions_checkpoint.json`
- Each checkpoint entry has `{"entities": [...], "relations": [...]}`

### 5b. Graph Build (`ingestion/core/graph_builder.py`)
- Creates nodes in Neo4j for each entity (companies, concepts, financial items)
- Creates relationships between entities based on extracted relations
- Clears existing graph on first filing (controlled by `clear_graph=True`)

### 5c. Community Detection (`ingestion/core/community_detector.py`)
- Runs Louvain algorithm on the Neo4j graph
- Detects communities of related entities
- Output: list of community assignments per entity

### 5d. Community Reports + Vector Index
- **Reports** (`ingestion/core/community_reports.py`): Generates summary text for each community using LLM
- **Vector Index** (`ingestion/core/vector_indexer.py`): Embeds entities + community reports into Neo4j vector index (`entity_embeddings`, 384 dimensions, cosine similarity)

**Output:** Neo4j graph persisted in Docker volume `prism_neo4j_data`

**Verification:**
```bash
# Neo4j Browser
open http://localhost:7474

# Or via cypher-shell
docker exec faithfulvoice_neo4j cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n)"
```

---

## Current State of the Data

| Artifact | Location | Status |
|---|---|---|
| Raw HTML filings | `data/raw/*.htm` | 35 files present |
| Cleaned HTML | `data/raw/*_clean.htm` | Most present |
| Chunk JSONL files | `data/processed/*_chunks.jsonl` | 32 files present |
| Qdrant vectors | Docker volume `prism_qdrant_data` | `prism_filings` collection with dense + sparse vectors |
| Neo4j graph | Docker volume `prism_neo4j_data` | Built from checkpoint |
| Graph checkpoints | `data/graphs/` | Entity extractions + community reports |
| Table images | `data/processed/tables/` | Rendered PNGs for financial tables |

**Known gaps:**
- Intel (`INTC`) raw filings exist but no processed chunk files were generated
- JPM query datasets exist but no JPM raw filings or chunks in the snapshot

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| Download filings | `python -m ingestion.scripts.download_filings --output-dir data/raw` |
| Process all filings (clean + chunk) | `python -m ingestion.scripts.process_all_filings` |
| Process only Intel filings | `python -m ingestion.scripts.process_all_filings --ticker INTC --force` |
| Load chunks into Qdrant | `python -m ingestion.scripts.ingest --recreate` |
| Build Neo4j graph from checkpoint | `python -m ingestion.scripts.run_neo4j_pipeline --skip-extraction` |
| Preview graph pipeline | `python -m ingestion.scripts.run_neo4j_pipeline --dry-run` |
| Start databases | `docker compose up -d qdrant neo4j` |
| Stop databases | `docker compose down` |

---

## Exporting and Sharing Qdrant Data

The Qdrant data lives in a Docker volume. To export it:

```bash
docker run --rm -v prism_qdrant_data:/volume -v $(pwd):/backup alpine tar czf /backup/prism_qdrant_data.tar.gz -C /volume .
```

To import on another machine:

```bash
docker volume create prism_qdrant_data
docker run --rm -v prism_qdrant_data:/volume -v /path/to/archive:/backup alpine tar xzf /backup/prism_qdrant_data.tar.gz -C /volume .
docker compose up -d qdrant
```
