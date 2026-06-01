# SEC Filing Chunking & Retrieval Strategy

> Why plain RAG fails on SEC filings — and exactly what to do instead.  
> **This strategy is optimised for EDGAR HTML filings (HTM format), which are preferred over PDF.**

---

## Why SEC Filings Break Naive RAG

SEC filings (10-K, 10-Q) are some of the worst documents for standard chunking:

| Problem | Impact on RAG |
| --------------------------------------------------------------- | --------------------------------------------------------------------- |
| Dense financial tables (balance sheets, income statements) | Fixed-size chunking cuts tables mid-row, producing garbage embeddings |
| Thousands of tokens per section (MD&A alone can be 20k+ tokens) | Context overflow, lost in the middle |
| Repeated boilerplate (legal disclaimers, signatures) | Pollutes retrieval with irrelevant noise |
| Numerical precision matters ("$2.3B" vs "$23B") | Dense embeddings are bad at exact numbers |
| Hierarchical structure (Item 1A → sub-sections → footnotes) | Flat chunking destroys hierarchy |
| Inline iXBRL tags wrapping every financial value | Dirty text confuses embeddings — must be stripped before parsing |

**The fix is a multi-layer strategy: pre-clean XBRL, parse by HTML structure, chunk by element type, retrieve with hybrid search.**

---

## Why HTML over PDF

EDGAR HTML filings are superior to PDF for RAG in every measurable way:

| Dimension | PDF | HTML (EDGAR HTM) |
| -------------------- | ---------------------------------------- | ------------------------------------------------- |
| Parsing speed | Slow — requires `strategy="hi_res"` OCR | ~10× faster — no OCR needed |
| Table extraction | Fragile — layout-based heuristics | Reliable — `pd.read_html()` reads DOM directly |
| Section detection | Page-number heuristics, brittle | **Text-based detection (see below)** |
| Text cleanliness | OCR artifacts, ligature errors | Clean Unicode text |
| Structure awareness | None — flat page stream | Section spans preserved by `unstructured` |
| Source deep-linking | Page number only | `accession_number` + `anchor_id` → direct SEC URL |

**Always download the primary 10-K HTM document from the EDGAR full submission package, not the PDF.**

---

## Layer 0 — Pre-Cleaning: Strip XBRL Before Parsing

EDGAR HTML filings embed iXBRL (Inline XBRL) tags that wrap every reported financial figure. These produce dirty element text like `<ix:nonfraction ...>15,234</ix:nonfraction>` which corrupts `unstructured` output. Strip them first.

```python
from bs4 import BeautifulSoup
import re

def clean_edgar_html(filepath: str) -> str:
    """Remove iXBRL tags and hidden XBRL context blocks from EDGAR HTML."""
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    # Unwrap all XBRL namespace tags — keeps inner text, removes tag wrapper
    for tag in soup.find_all(re.compile(r"^ix:")):
        tag.unwrap()

    # Decompose hidden XBRL schema/context blocks entirely
    for tag in soup.find_all("div", style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()

    # Decompose the ix:header block (contains XBRL context metadata)
    ix_header = soup.find("ix:header")
    if ix_header:
        ix_header.decompose()

    cleaned_path = filepath.replace(".htm", "_clean.htm")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    return cleaned_path
```

**This step is mandatory** — skip it and every financial figure in your embeddings will be wrapped in garbage XML.

---

## Layer 1 — Parsing: Use `unstructured` on Cleaned HTML

After pre-cleaning, pass the cleaned HTM to `unstructured`. Do **not** use `PyPDF2`, raw `BeautifulSoup`, or `partition_pdf` — they miss document structure.

```python
from unstructured.partition.html import partition_html

# Always use partition_html — not partition_pdf — for EDGAR filings
elements = partition_html(
    filename=cleaned_path,
    skip_headers_and_footers=True,
    include_metadata=True,
)
```

`unstructured` returns typed elements:

| Element Type | What it is |
| ------------------- | ------------------------------------------------- |
| `NarrativeText` | Prose paragraphs (most common) |
| `Text` | Short text blocks, headers |
| `Table` | Detected table — supplement with `pd.read_html()` |
| `ListItem` | Bullet/numbered list items |
| `Header` / `Footer` | Page metadata — discard |
| `PageBreak` | Structural marker — discard |
| `Title` | **Rarely produced** — don't rely on it |

> **Critical Finding:** Modern iXBRL filings produce **0 Title elements**. Section headers are `<span>` tags with inline styles, not `<h1>`-`<<h6>` tags.

---

## Section Detection: Text-Based (NOT Anchor-Based)

> **The docs assumed `<a name="itemX">` anchors exist — they don't.** Modern iXBRL filings use a completely different structure.

### What Actually Happens

Modern EDGAR HTM filings use this structure:

1. **TOC links:** `<a href="#i13eac97307cc485c971e826acbda8be7_13">Item 1.</a>` (inside `<span>`)
2. **Section markers:** `<div id="i13eac97307cc485c971e826acbda8be7_13">`  — empty anchor markers
3. **Actual headers:** `<span style="color:#76b900">Item 1A. Risk Factors</span>` (NVIDIA green) or `<span style="color:#0000ff">Item 1A.</span>` (blue)

### How to Detect Sections (Correct Approach)

```python
import re

SECTION_PATTERNS = [
    (re.compile(r"^item\s*1\b", re.I), "item1", "Business Overview"),
    (re.compile(r"^item\s*1a\b", re.I), "item1a", "Risk Factors"),
    (re.compile(r"^item\s*1b\b", re.I), "item1b", "Unresolved Staff Comments"),
    (re.compile(r"^item\s*1c\b", re.I), "item1c", "Cybersecurity"),
    (re.compile(r"^item\s*2\b", re.I), "item2", "Properties"),
    (re.compile(r"^item\s*3\b", re.I), "item3", "Legal Proceedings"),
    (re.compile(r"^item\s*7\b", re.I), "item7", "MD&A"),
    (re.compile(r"^item\s*7a\b", re.I), "item7a", "Market Risk"),
    (re.compile(r"^item\s*8\b", re.I), "item8", "Financial Statements"),
    (re.compile(r"^item\s*9a\b", re.I), "item9a", "Controls and Procedures"),
    (re.compile(r"^item\s*15\b", re.I), "item15", "Exhibits and Schedules"),
    # ... add all 16 Items as needed
]

BOILERPLATE_PATTERNS = [
    re.compile(r"pursuant to the requirements of the securities exchange act", re.I),
    re.compile(r"incorporated herein by reference", re.I),
    re.compile(r"see exhibit index", re.I),
    re.compile(r"^table of contents$", re.I),
    re.compile(r"^item\s*\d+[a-z]?\.$", re.I),  # TOC stub entries
]

def _detect_section(text: str) -> tuple[str, str] | None:
    """Returns (anchor_id, section_label) if text matches an Item pattern."""
    for pattern, anchor_id, section_label in SECTION_PATTERNS:
        if pattern.search(text.strip()):
            return anchor_id, section_label
    return None

def should_skip(text: str) -> bool:
    """Filter out boilerplate and short junk."""
    text = text.strip()
    if len(text) < 50:
        return True
    if re.match(r"^\$?[\d,\.]+[bmk]?$", text):  # XBRL numeric artifact
        return True
    return any(p.search(text) for p in BOILERPLATE_PATTERNS)
```

### How It Works in Practice

1. Iterate through `unstructured` elements in order
2. For each element's text, check if it matches `"Item X."` pattern
3. If match → update `current_anchor_id` and `current_section`
4. Stamp all subsequent elements with current anchor_id until a new match

---

## Layer 2 — Chunking Strategy: Route by Element Type

> **Key insight from research:** Element-type-based chunking outperforms paragraph-level chunking on financial documents. Structure is signal.

### 2a. Text Chunks — Parent-Child Recursive Splitting

For `NarrativeText` elements, use a **parent-child chunking** approach:

- **Parent chunk**: ~2,000 tokens — used for context window injection
- **Child chunk**: ~512 tokens — used for embedding & retrieval

Why? Small chunks embed with higher precision. But at retrieval time you return the _parent_ for more context.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=512,  chunk_overlap=100)
```

**Skip these sections** (mostly boilerplate/exhibits):

```python
SKIP_ANCHORS = {
    "cover",      # Cover page
    "item4",     # Mine Safety
    "item6",     # Reserved
    "item9",     # Accountant Disagreements
    "item9b",    # Other Information
    "item9c",    # Foreign Jurisdiction
    "item10",     # Directors/Officers
    "item11",     # Executive Compensation
    "item14",     # Principal Accountant Fees
    "item16",     # 10-K Summary
}
```

### 2b. Table Chunks — Dual Representation

Tables are the hardest problem. Strategy:

1. Extract table text from `unstructured` Table element
2. Try `pd.read_html()` on raw HTML for markdown
3. Store: `text` (summary/ellipsis) + `table_markdown` (raw, for context)

```python
import pandas as pd

def extract_table_markdown(table_element_text: str) -> str:
    """Try to extract markdown from table element text."""
    try:
        # table_element_text may contain HTML table
        if "<table" in table_element_text:
            dfs = pd.read_html(table_element_text, flavor="lxml")
            if dfs:
                return dfs[0].to_markdown(index=False)
    except Exception:
        pass
    return ""  # Return empty if no table found
```

### 2c. Metadata — Attach to Every Chunk

```python
{
    "text": "...",                    # chunk text (child for embedding)
    "parent_text": "...",             # parent context (for LLM injection)
    "table_markdown": "...",         # raw table markdown (tables only)
    "company": "NVIDIA Corporation",
    "ticker": "NVDA",
    "cik": "0001045810",
    "year": 2024,
    "quarter": None,               # None for 10-K, 1-4 for 10-Q
    "filing_type": "10K",
    "section": "Risk Factors",
    "anchor_id": "item1a",       # ← PRIMARY FILTER KEY
    "element_type": "NarrativeText",  # or "Table"
    "chunk_index": 4,
    "parent_chunk_id": "abc123",
    "chunk_id": "def456",
    "source_url": "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm#item1a",
    "htm_filename": "nvda-20240128.htm"
}
```

---

## Layer 3 — Embedding: Finance-Aware Models

| Model | Dims | Notes |
| ----------------------------------------- | ---- | -------------------------------------------------- |
| `BAAI/bge-base-en-v1.5` | 768 | Strong MTEB scores, good finance performance, free |
| `intfloat/e5-base-v2` | 768 | Solid all-rounder, handles financial text well |
| `text-embedding-3-large` (OpenAI) | 3072 | Best quality, costs money, good for demo polish |

```python
from fastembed import TextEmbedding, SparseTextEmbedding

dense_model  = TextEmbedding("BAAI/bge-base-en-v1.5")
sparse_model = SparseTextEmbedding("Qdrant/bm25")
```

---

## Layer 4 — Qdrant Schema: Multi-Vector Collection

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, Modifier

client = QdrantClient("localhost", port=6333)

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

---

## Layer 5 — Retrieval: Hybrid Search with Anchor-Filtered RRF

```python
from qdrant_client.models import Prefetch, FusionQuery, Fusion, Filter, FieldCondition, MatchValue, SparseVector

INTENT_TO_ANCHOR = {
    "risk_factors":      "item1a",
    "business_overview": "item1",
    "financial_data":    "item8",
    "mda":           "item7",
    "market_risk":     "item7a",
    "controls":       "item9a",
}

def hybrid_search(query: str, ticker: str = None, anchor_id: str = None, top_k: int = 8):
    conditions = []
    if ticker:
        conditions.append(FieldCondition(key="ticker", match=MatchValue(value=ticker)))
    if anchor_id:
        conditions.append(FieldCondition(key="anchor_id", match=MatchValue(value=anchor_id)))

    query_filter = Filter(must=conditions) if conditions else None
    dense_q  = list(dense_model.embed([query]))[0].tolist()
    sparse_q = list(sparse_model.embed([query]))[0]

    results = client.query_points(
        collection_name="sec_filings",
        prefetch=[
            Prefetch(query=SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist()), using="sparse", limit=20),
            Prefetch(query=dense_q, using="dense", limit=20)
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=query_filter,
        limit=top_k
    )
    return [r.payload for r in results.points]
```

---

## How to Run: Complete Pipeline

### Prerequisites

```bash
# Create conda environment
conda create -n faithfulvoice python=3.11
conda activate faithfulvoice

# Install dependencies
pip install qdrant-client langchain-text-splitters \
    beautifulsoup4 lxml pandas requests \
    fastembed unstructured
```

### Step 1: Download Filings

```bash
# Download all filings
python -m ingestion.scripts.download_filings

# Or specific tickers
python -m ingestion.scripts.download_filings --tickers NVDA AMD INTC
```

Output: `data/raw/NVDA_2024_10K.htm`, etc. (36 files, ~120 MB total)

### Step 2: Clean + Parse + Chunk (One Filing)

```python
from src.cleaner import clean_edgar_html
from src.chunker import chunk_filing
from dataclasses import asdict

# Single filing pipeline
raw_path = "data/raw/NVDA_2024_10K.htm"
cleaned_path = clean_edgar_html(raw_path)
chunks = chunk_filing(cleaned_path, raw_path)

# Write to JSONL
import json
with open("data/processed/NVDA_2024_10K_chunks.jsonl", "w") as f:
    for c in chunks:
        f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
```

### Step 3: Batch Process All Filings

```python
from pathlib import Path
from src.cleaner import clean_edgar_html
from src.chunker import chunk_filing
import json

raw_dir = Path("data/raw")
processed_dir = Path("data/processed")
processed_dir.mkdir(exist_ok=True)

for htm_file in raw_dir.glob("*_10K.htm"):
    print(f"Processing {htm_file.name}...")
    cleaned = clean_edgar_html(htm_file)
    chunks = chunk_filing(cleaned, htm_file)
    
    out_file = processed_dir / f"{htm_file.stem}_chunks.jsonl"
    with open(out_file, "w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"  → {len(chunks)} chunks")
```

### Step 4: Embed + Upsert to Qdrant

```python
# (See Layer 4 code above)
# Batch embed all chunks, upsert to Qdrant collection
```

---

## What NOT to Do

| ❌ Don't | ✅ Do instead |
| ------------------------------------------------- | ---------------------------------------------------------- |
| Use `partition_pdf` on EDGAR filings | Use `partition_html` on the primary HTM document |
| Parse raw HTM without stripping XBRL first | Run `clean_edgar_html()` before any parsing |
| **Look for `<a name="itemX">` anchors** | **Use text-based section detection** (see Layer 1) |
| Rely on Title elements from `unstructured` | **Match "Item X." patterns in element text** |
| Use only dense vector search | Hybrid: dense + BM25 with RRF |
| Filter only by company/year | Add `anchor_id` filter — highest-precision filter |
| Use `all-MiniLM-L6-v2` for finance | Use `BAAI/bge-base-en-v1.5` minimum |
| Ignore `anchor_id` in metadata | Store it — enables section filtering |

---

## Summary Cheatsheet

```
Raw EDGAR HTM (data/raw/NVDA_2024_10K.htm)
    → clean_edgar_html()              # strip ix:* tags, ix:header, hidden divs
    → partition_html()              # unstructured element extraction
    → text-based section detection (regex "Item X.")
    → element-type routing:
         NarrativeText → recursive split (512 child / 2000 parent) + anchor_id stamp
         Table      → dual rep (text + markdown)
         short/junk → filtered by should_skip()
    → metadata tagging (ticker, year, cik, anchor_id, source_url, ...)
    → Qdrant upsert: dense (bge-base) + sparse (BM25)

Query
    → Query Analyzer: extract ticker → map intent to anchor_id
    → Qdrant hybrid search: RRF, filtered by ticker + anchor_id
    → Relevance Grader: drop irrelevant chunks
    → Context assembly: inject parent chunks + table markdown
    → LLM Answer: cited with section label + EDGAR deep-link URL
```
