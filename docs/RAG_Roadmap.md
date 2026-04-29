# 🗺️ SEC Filing Research Agent — Project Roadmap

> **Stack:** LangGraph · Qdrant · `unstructured` · `sentence-transformers` / `FinLang` · Python  
> **Timeline:** 2 days (aggressive but doable for a demo)  
> **Pattern:** Corrective RAG (CRAG) over SEC 10-K / 10-Q filings

---

## Phase 0 — Environment Setup _(~1 hour)_

Get everything running locally before touching data.

### Goals

- Docker Qdrant running locally
- Python virtual env with all deps
- EDGAR filings downloaded (3–5 companies, 10-K)

### Steps

```bash
# Qdrant in Docker
docker pull qdrant/qdrant
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Python deps
pip install qdrant-client langgraph langchain langchain-openai \
    unstructured[pdf,html] sentence-transformers openai \
    fastembed pandas rich python-dotenv
```

### EDGAR Filing Download

Go to https://www.sec.gov/cgi-bin/browse-edgar → pick 3–5 companies → download 10-K HTML or PDF.  
Suggested: Apple (AAPL), Microsoft (MSFT), JPMorgan (JPM) — mix of tech + finance.

### Deliverable

- `docker ps` shows Qdrant at port 6333
- `data/raw/` folder with 3–5 filings

---

## Phase 1 — Document Parsing & Intelligent Chunking _(~3 hours, Day 1)_

This is the hardest and most important phase. SEC filings are not clean text — they have dense tables, XBRL tags, nested HTML, footnotes. Plain `split("\n")` chunking will destroy context. See [`CHUNKING_STRATEGY.md`](./CHUNKING_STRATEGY.md) for full detail.

### Goals

- Parse raw filings into structured elements (text, tables, titles)
- Chunk by element type, not fixed token windows
- Generate table summaries via LLM
- Attach rich metadata to every chunk

### Pipeline

```
Raw HTML/PDF
    ↓
unstructured.partition_html() / partition_pdf()
    ↓  (returns: NarrativeText, Table, Title, ListItem, ...)
Element-type routing:
    - NarrativeText → section-aware recursive chunking (~512 tokens, 100 overlap)
    - Table         → LLM summary + raw markdown preserved
    - Title         → used as parent context header only
    ↓
Metadata enrichment:
    { company, ticker, year, filing_type, section, element_type, page }
    ↓
Chunk store: List[dict]
```

### Key Code Files

- `src/parser.py` — wraps `unstructured`, routes element types
- `src/chunker.py` — section-aware chunking + table summarization
- `src/metadata.py` — metadata extraction from filename + filing header

### Deliverable

- `data/processed/chunks.jsonl` — all chunks with metadata
- Quick stats: how many chunks per filing, per section, per element type

---

## Phase 2 — Qdrant Ingestion with Hybrid Indexing _(~2 hours, Day 1)_

Store chunks with **both dense and sparse vectors** so retrieval can combine semantic + keyword search. See [`CHUNKING_STRATEGY.md`](./CHUNKING_STRATEGY.md) for full rationale.

### Goals

- Create Qdrant collection with named dense + sparse vector configs
- Embed text chunks using a finance-aware embedding model
- Store table summaries as dense; raw table markdown in payload
- Insert all chunks with full metadata as Qdrant payload

### Collection Schema

```python
client.create_collection(
    collection_name="sec_filings",
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(modifier=Modifier.IDF)
    }
)
```

### Payload per Point

```json
{
  "text": "...",
  "table_markdown": "...", // only for table chunks
  "summary": "...", // LLM-generated, for table chunks
  "company": "Apple Inc.",
  "ticker": "AAPL",
  "year": 2023,
  "filing_type": "10-K",
  "section": "Risk Factors",
  "element_type": "NarrativeText | Table",
  "page": 14
}
```

### Key Code Files

- `src/embedder.py` — embedding model wrapper (dense + sparse via FastEmbed)
- `src/ingest.py` — batch upsert to Qdrant

### Deliverable

- Qdrant collection `sec_filings` populated
- Sanity check: 5 manual queries returning expected chunks

---

## Phase 3 — LangGraph CRAG Agent _(~4 hours, Day 2)_

Build the multi-node graph that drives the Q&A loop. The pattern is **Corrective RAG**: retrieve → grade → answer or retry.

### Graph Nodes

```
[User Query]
     ↓
┌─────────────────┐
│  Query Analyzer │  → classifies intent: financial? risk? operations?
│                 │    also generates query expansion / HyDE hypothesis
└────────┬────────┘
         ↓
┌─────────────────┐
│    Retriever    │  → Hybrid search (dense + BM25 + RRF fusion)
│                 │    Qdrant prefetch + FusionQuery(RRF)
│                 │    Filter by: company, year, section (from metadata)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Relevance      │  → LLM grades each retrieved chunk (relevant / not)
│  Grader         │    Drops irrelevant chunks
└────────┬────────┘
     ↙       ↘
[Good]      [Bad / Empty]
  ↓               ↓
┌──────┐    ┌──────────────┐
│Answer│    │ Query Rewriter│ → rewrites query → back to Retriever
│ Node │    └──────────────┘
└──────┘
```

### Node Implementations

**Query Analyzer Node**

```python
# Classifies: {"intent": "risk_factors", "entities": ["AAPL", "2023"], "section_hint": "Risk Factors"}
# Generates HyDE: a hypothetical passage the answer might appear in
```

**Retriever Node**

```python
client.query_points(
    collection_name="sec_filings",
    prefetch=[
        Prefetch(query=sparse_vec, using="sparse", limit=20),
        Prefetch(query=dense_vec, using="dense", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    query_filter=Filter(must=[...company/year conditions...]),
    limit=8
)
```

**Grader Node**

```python
# Prompt: "Is this chunk relevant to the question? Return YES or NO."
# Filters chunk list, keeps only YES
```

**Answer Node**

```python
# Prompt with all relevant chunks as context
# Produces cited answer: "According to Apple's 2023 10-K, Risk Factors section..."
```

**Query Rewriter Node**

```python
# If grader found 0 relevant chunks: rewrite query, signal retry
# Max 2 retries to avoid infinite loop
```

### State Object

```python
class AgentState(TypedDict):
    question: str
    query_analysis: dict
    retrieved_chunks: list
    graded_chunks: list
    answer: str
    retry_count: int
```

### Key Code Files

- `src/graph.py` — LangGraph StateGraph definition
- `src/nodes/` — one file per node
- `src/retriever.py` — Qdrant hybrid search wrapper

### Deliverable

- `python src/graph.py "What are Apple's main risk factors in 2023?"` returns a cited answer
- Graph trace visible (LangGraph's built-in logging)

---

## Phase 4 — Demo UI _(~1 hour, Day 2)_

Wrap the agent in a simple Streamlit interface for demo purposes.

### Goals

- Text input for question
- Sidebar: filter by company, year, section
- Answer panel with source citations
- Show retrieved chunks (expandable)

### Key Code Files

- `app.py` — Streamlit entrypoint

### Deliverable

- `streamlit run app.py` shows working UI
- Can demo: type question → see graph reasoning → see answer with citations

---

## Phase 5 — Demo Polish _(~1 hour, Day 2 end)_

Make it presentable.

- [ ] Add 3–5 canned "example questions" in the UI
- [ ] Log graph trace steps visibly (Retriever → Grader → Answer path)
- [ ] Handle edge cases: no results found, query out of scope
- [ ] Add a `README.md` with setup instructions

---

## File Structure

```
sec-agent/
├── data/
│   ├── raw/             # Downloaded EDGAR filings
│   └── processed/       # chunks.jsonl
├── src/
│   ├── parser.py
│   ├── chunker.py
│   ├── metadata.py
│   ├── embedder.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── graph.py
│   └── nodes/
│       ├── analyzer.py
│       ├── retriever_node.py
│       ├── grader.py
│       ├── answer.py
│       └── rewriter.py
├── app.py               # Streamlit UI
├── .env                 # OPENAI_API_KEY etc.
├── ROADMAP.md
├── CHUNKING_STRATEGY.md
└── requirements.txt
```

---

## Day-by-Day Summary

| Time     | Task                                |
| -------- | ----------------------------------- |
| Day 1 AM | Phase 0 + Phase 1 (setup + parsing) |
| Day 1 PM | Phase 2 (Qdrant ingestion + verify) |
| Day 2 AM | Phase 3 (LangGraph agent)           |
| Day 2 PM | Phase 4 + 5 (UI + polish)           |

---

## Demo Questions to Prepare

1. _"What are Apple's main risk factors in their 2023 10-K?"_
2. _"How did Microsoft's revenue change year-over-year?"_
3. _"What does JPMorgan say about credit risk exposure?"_
4. _"Compare liquidity disclosures between Apple and Microsoft."_
5. _"What is Apple's capital allocation strategy?"_
