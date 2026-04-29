# 🗺️ SEC Filing Research Agent — Project Roadmap

> **Stack:** LangGraph · Qdrant · `unstructured` · `sentence-transformers` / `FinLang` · Python  
> **Timeline:** 2 days (aggressive but doable for a demo)  
> **Pattern:** Corrective RAG (CRAG) over SEC 10-K / 10-Q filings  
> **Format:** HTML (EDGAR HTM filings — preferred over PDF for accuracy and speed)

---

## Phase 0 — Environment Setup _(~1 hour)_

Get everything running locally before touching data.

### Goals

- Docker Qdrant running locally
- Python virtual env with all deps
- EDGAR filings downloaded (3–5 companies, 10-K) as HTML packages

### Steps

```bash
# Qdrant in Docker
docker pull qdrant/qdrant
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Python deps
pip install qdrant-client langgraph langchain langchain-openai \
    unstructured[html] sentence-transformers openai \
    fastembed pandas lxml beautifulsoup4 rich python-dotenv requests
```

> Note: `unstructured[html]` replaces `unstructured[pdf,html]` — no PDF pipeline needed.

### EDGAR Filing Download

EDGAR filings come as full submission packages (ZIP with multiple HTM files). Do **not** just download a single PDF — the HTML package is richer and cleaner.

```python
import requests, json

def get_filing_package(cik: str, accession_number: str):
    """Fetch the full submission index for a filing."""
    acc_clean = accession_number.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{accession_number}-index.json"
    index = requests.get(index_url, headers={"User-Agent": "yourname@email.com"}).json()
    
    # Only process the primary 10-K document — skip exhibits (EX-*) and XBRL viewer (R*.htm)
    primary = [f for f in index["directory"]["item"]
               if f["type"] == "10-K" and not f["name"].startswith("R")]
    return primary
```

Suggested companies: Apple (AAPL, CIK 320193), Microsoft (MSFT, CIK 789019), JPMorgan (JPM, CIK 19617) — mix of tech + finance.

### Deliverable

- `docker ps` shows Qdrant at port 6333
- `data/raw/` folder with 3–5 `.htm` filing files (primary 10-K documents only)

---

## Phase 1 — Document Parsing & Intelligent Chunking _(~3 hours, Day 1)_

This is the hardest and most important phase. SEC HTML filings contain dense tables, inline XBRL tags, and hierarchical Item structure. Plain `split("\n")` chunking will destroy context. See [`CHUNKING_STRATEGY.md`](./CHUNKING_STRATEGY.md) for full detail.

### Goals

- Pre-clean XBRL inline tags from raw HTML before parsing
- Parse cleaned HTML into structured elements (text, tables, titles)
- Detect sections using EDGAR's `<a name="itemX">` anchors — more reliable than page heuristics
- Chunk by element type, not fixed token windows
- Extract tables via `pandas.read_html()` for complex nested tables
- Generate table summaries via LLM
- Attach rich metadata (including `anchor_id`, `accession_number`, `source_url`) to every chunk

### Pre-Cleaning Pipeline

SEC EDGAR HTML filings use inline iXBRL tags that wrap financial values. These must be stripped before `unstructured` sees the file, or they produce dirty element text.

```python
from bs4 import BeautifulSoup
import re

def clean_edgar_html(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    # Strip XBRL namespace tags (ix:nonfraction, ix:nonnumeric, etc.) but keep inner text
    for tag in soup.find_all(re.compile(r"^ix:")):
        tag.unwrap()

    # Remove hidden XBRL context/schema blocks entirely
    for tag in soup.find_all("div", style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()

    # Save cleaned HTML for unstructured
    cleaned_path = filepath.replace(".htm", "_clean.htm")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    return cleaned_path
```

### Parsing & Section Detection Pipeline

```
Raw EDGAR HTM
    ↓
clean_edgar_html()          ← strips XBRL tags, hidden divs
    ↓
extract_sections_from_html() ← maps <a name="item1a"> → "Item 1A. Risk Factors"
    ↓
partition_html(cleaned_path, skip_headers_and_footers=True)
    ↓  (returns: NarrativeText, Table, Title, ListItem, ...)
Element-type routing:
    - NarrativeText → section-aware recursive chunking (~512 tokens child, ~2000 parent)
    - Table         → pd.read_html() extraction → LLM summary + raw markdown preserved
    - Title         → updates current_section tracker (anchor-based when available)
    - Header/Footer → discard
    ↓
Metadata enrichment:
    { company, ticker, year, filing_type, section, anchor_id,
      element_type, chunk_index, parent_chunk_id,
      accession_number, source_url, cik, htm_filename }
    ↓
Chunk store: List[dict]
```

**Section detection using HTML anchors:**

```python
SECTION_MAP = {
    "item1":  "Business Overview",
    "item1a": "Risk Factors",
    "item1b": "Unresolved Staff Comments",
    "item2":  "Properties",
    "item7":  "MD&A",
    "item7a": "Market Risk",
    "item8":  "Financial Statements",
    "item9a": "Controls and Procedures",
}

def extract_anchor_sections(soup) -> dict:
    """Returns {anchor_id: human_label} from EDGAR <a name="..."> tags."""
    sections = {}
    for anchor in soup.find_all("a", attrs={"name": True}):
        key = anchor["name"].lower().replace(" ", "")
        if key in SECTION_MAP:
            sections[key] = SECTION_MAP[key]
    return sections
```

### Key Code Files

- `src/cleaner.py` — XBRL stripping and hidden-element removal
- `src/parser.py` — wraps `unstructured`, anchor-based section detection, element-type routing
- `src/chunker.py` — section-aware chunking + `pd.read_html()` table extraction + LLM summarization
- `src/metadata.py` — metadata assembly (CIK, accession number, source URL, anchor ID)

### Deliverable

- `data/processed/chunks.jsonl` — all chunks with full metadata
- Quick stats: chunks per filing, per section (`anchor_id`), per element type

---

## Phase 2 — Qdrant Ingestion with Hybrid Indexing _(~2 hours, Day 1)_

Store chunks with **both dense and sparse vectors** so retrieval can combine semantic + keyword search. The expanded metadata schema (with `anchor_id`) also enables precise section-level filtering at query time. See [`CHUNKING_STRATEGY.md`](./CHUNKING_STRATEGY.md) for full rationale.

### Goals

- Create Qdrant collection with named dense + sparse vector configs
- Embed text chunks using a finance-aware embedding model
- Store table summaries as the dense embedding; raw table markdown in payload
- Insert all chunks with full metadata (including `anchor_id`, `source_url`) as Qdrant payload

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
  "table_markdown": "...",          // only for table chunks — extracted via pd.read_html()
  "summary": "...",                 // LLM-generated, for table chunks — this is what gets embedded
  "company": "Apple Inc.",
  "ticker": "AAPL",
  "cik": "320193",
  "year": 2023,
  "filing_type": "10-K",
  "accession_number": "0000320193-23-000106",
  "section": "Risk Factors",
  "anchor_id": "item1a",            // EDGAR HTML anchor — enables exact section filtering
  "element_type": "NarrativeText | Table",
  "chunk_index": 4,
  "parent_chunk_id": "abc123",
  "source_url": "https://www.sec.gov/Archives/edgar/data/320193/.../aapl-20230930.htm",
  "htm_filename": "aapl-20230930.htm"
}
```

> `accession_number` + `anchor_id` together reconstruct a direct deep-link to the exact section in the SEC EDGAR viewer — invaluable for answer citations.

### Key Code Files

- `src/embedder.py` — embedding model wrapper (dense + sparse via FastEmbed)
- `src/ingest.py` — batch upsert to Qdrant

### Deliverable

- Qdrant collection `sec_filings` populated
- Sanity check: 5 manual queries returning expected chunks with correct `anchor_id` values

---

## Phase 3 — LangGraph CRAG Agent _(~4 hours, Day 2)_

Build the multi-node graph that drives the Q&A loop. The pattern is **Corrective RAG**: retrieve → grade → answer or retry. The Query Analyzer now also maps user intent to `anchor_id` for hard section filtering before semantic search.

### Graph Nodes

```
[User Query]
     ↓
┌─────────────────┐
│  Query Analyzer │  → classifies intent: financial? risk? operations?
│                 │    maps intent → anchor_id (e.g. "risk" → "item1a")
│                 │    extracts entities (ticker, year)
│                 │    generates HyDE hypothesis for qualitative queries
└────────┬────────┘
         ↓
┌─────────────────┐
│    Retriever    │  → Hybrid search (dense + BM25 + RRF fusion)
│                 │    Qdrant prefetch + FusionQuery(RRF)
│                 │    Filter by: ticker, year, anchor_id (hard metadata filter)
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
│ Node │    └──────────────┘   (drops anchor_id filter on retry)
└──────┘
```

### Node Implementations

**Query Analyzer Node**

```python
INTENT_TO_ANCHOR = {
    "risk_factors":      "item1a",
    "business_overview": "item1",
    "financial_data":    "item8",
    "mda":               "item7",
    "market_risk":       "item7a",
    "controls":          "item9a",
}

# Classifies: {
#   "intent": "risk_factors",
#   "anchor_id": "item1a",          ← maps intent to EDGAR section anchor
#   "entities": ["AAPL", "2023"],
#   "is_qualitative": True          ← determines whether to use HyDE
# }
# If qualitative: generate HyDE hypothesis passage
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
    query_filter=Filter(must=[
        FieldCondition(key="ticker", match=MatchValue(value="AAPL")),
        FieldCondition(key="year",   match=MatchValue(value=2023)),
        FieldCondition(key="anchor_id", match=MatchValue(value="item1a")),  # ← anchor filter
    ]),
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
# Cites using: company, year, section label, and direct EDGAR URL
# e.g. "According to Apple's 2023 10-K, Risk Factors (Item 1A)..."
# Source URL reconstructed from accession_number + anchor_id in payload
```

**Query Rewriter Node**

```python
# If grader found 0 relevant chunks:
#   - First retry: rewrite query, keep anchor_id filter
#   - Second retry: rewrite query, DROP anchor_id filter (broaden scope)
# Max 2 retries to avoid infinite loop
```

### State Object

```python
class AgentState(TypedDict):
    question: str
    query_analysis: dict       # includes anchor_id, entities, is_qualitative
    retrieved_chunks: list
    graded_chunks: list
    answer: str
    retry_count: int
```

### Key Code Files

- `src/graph.py` — LangGraph StateGraph definition
- `src/nodes/` — one file per node
- `src/retriever.py` — Qdrant hybrid search wrapper (with anchor_id filter support)

### Deliverable

- `python src/graph.py "What are Apple's main risk factors in 2023?"` returns a cited answer with source URL
- Graph trace visible (LangGraph's built-in logging)

---

## Phase 4 — Demo UI _(~1 hour, Day 2)_

Wrap the agent in a simple Streamlit interface for demo purposes.

### Goals

- Text input for question
- Sidebar: filter by company, year, section (mapped to `anchor_id` values)
- Answer panel with source citations (including clickable EDGAR deep-links)
- Show retrieved chunks (expandable, with section label + anchor)

### Key Code Files

- `app.py` — Streamlit entrypoint

### Deliverable

- `streamlit run app.py` shows working UI
- Citations include clickable links to exact SEC EDGAR filing sections

---

## Phase 5 — Demo Polish _(~1 hour, Day 2 end)_

Make it presentable.

- [ ] Add 3–5 canned "example questions" in the UI
- [ ] Log graph trace steps visibly (Analyzer → Retriever → Grader → Answer path)
- [ ] Show which `anchor_id` filter was applied per query
- [ ] Handle edge cases: no results found, query out of scope, anchor filter loosened on retry
- [ ] Add a `README.md` with setup instructions

---

## File Structure

```
sec-agent/
├── data/
│   ├── raw/             # Downloaded EDGAR .htm filings (primary 10-K docs only)
│   └── processed/       # chunks.jsonl
├── src/
│   ├── cleaner.py       # XBRL stripping + hidden-element removal  ← NEW
│   ├── parser.py        # unstructured wrapper + anchor-based section detection
│   ├── chunker.py       # section-aware chunking + pd.read_html() + LLM table summaries
│   ├── metadata.py      # metadata assembly (CIK, accession, anchor_id, source_url)
│   ├── embedder.py
│   ├── ingest.py
│   ├── retriever.py     # hybrid search with anchor_id filter support
│   ├── graph.py
│   └── nodes/
│       ├── analyzer.py  # intent → anchor_id mapping + HyDE
│       ├── retriever_node.py
│       ├── grader.py
│       ├── answer.py    # citations with EDGAR deep-links
│       └── rewriter.py  # drops anchor_id on second retry
├── app.py               # Streamlit UI
├── .env                 # OPENAI_API_KEY etc.
├── ROADMAP.md
├── CHUNKING_STRATEGY.md
└── requirements.txt
```

---

## Day-by-Day Summary

| Time     | Task                                                          |
| -------- | ------------------------------------------------------------- |
| Day 1 AM | Phase 0 + Phase 1 (setup + XBRL cleaning + HTML parsing)     |
| Day 1 PM | Phase 2 (Qdrant ingestion with anchor_id metadata + verify)  |
| Day 2 AM | Phase 3 (LangGraph agent with anchor-filtered retrieval)      |
| Day 2 PM | Phase 4 + 5 (UI with EDGAR deep-links + polish)              |

---

## Demo Questions to Prepare

1. _"What are Apple's main risk factors in their 2023 10-K?"_ → anchor filter: `item1a`
2. _"How did Microsoft's revenue change year-over-year?"_ → anchor filter: `item8`
3. _"What does JPMorgan say about credit risk exposure?"_ → anchor filter: `item7`
4. _"Compare liquidity disclosures between Apple and Microsoft."_ → anchor filter: `item7`
5. _"What is Apple's capital allocation strategy?"_ → anchor filter: `item7` or `item8`
