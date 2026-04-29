# 📄 SEC Filing Chunking & Retrieval Strategy

> Why plain RAG fails on SEC filings — and exactly what to do instead.  
> **This strategy is optimised for EDGAR HTML filings (HTM format), which are preferred over PDF.**

---

## Why SEC Filings Break Naive RAG

SEC filings (10-K, 10-Q) are some of the worst documents for standard chunking:

| Problem                                                         | Impact on RAG                                                         |
| --------------------------------------------------------------- | --------------------------------------------------------------------- |
| Dense financial tables (balance sheets, income statements)      | Fixed-size chunking cuts tables mid-row, producing garbage embeddings |
| Thousands of tokens per section (MD&A alone can be 20k+ tokens) | Context overflow, lost in the middle                                  |
| Repeated boilerplate (legal disclaimers, signatures)            | Pollutes retrieval with irrelevant noise                              |
| Numerical precision matters ("$2.3B" vs "$23B")                 | Dense embeddings are bad at exact numbers                             |
| Hierarchical structure (Item 1A → sub-sections → footnotes)     | Flat chunking destroys hierarchy                                      |
| Inline iXBRL tags wrapping every financial value                | Dirty text confuses embeddings — must be stripped before parsing      |

**The fix is a multi-layer strategy: pre-clean XBRL, parse by HTML structure, chunk by element type, retrieve with hybrid search.**

---

## Why HTML over PDF

EDGAR HTML filings are superior to PDF for RAG in every measurable way:

| Dimension            | PDF                                      | HTML (EDGAR HTM)                                  |
| -------------------- | ---------------------------------------- | ------------------------------------------------- |
| Parsing speed        | Slow — requires `strategy="hi_res"` OCR | ~10× faster — no OCR needed                      |
| Table extraction     | Fragile — layout-based heuristics        | Reliable — `pd.read_html()` reads DOM directly   |
| Section detection    | Page-number heuristics, brittle          | `<a name="item1a">` anchors — exact and stable   |
| Text cleanliness     | OCR artifacts, ligature errors           | Clean Unicode text                                |
| Structure awareness  | None — flat page stream                  | `<h1>`–`<h4>` hierarchy preserved by `unstructured` |
| Source deep-linking  | Page number only                         | `accession_number` + `anchor_id` → direct SEC URL |

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

    cleaned_path = filepath.replace(".htm", "_clean.htm")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    return cleaned_path
```

This step is **mandatory** — skip it and every financial figure in your embeddings will be wrapped in garbage XML.

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

`unstructured` returns a list of typed elements. HTML structure means these are more accurate than PDF equivalents:

| Element Type        | What it is                                        |
| ------------------- | ------------------------------------------------- |
| `Title`             | Section header — maps to `<h1>`–`<h4>` in HTML   |
| `NarrativeText`     | Prose paragraphs                                  |
| `Table`             | Detected table — supplement with `pd.read_html()` |
| `ListItem`          | Bullet/numbered list items                        |
| `Header` / `Footer` | Page metadata — discard                           |
| `PageBreak`         | Structural marker — discard                       |

### Section Detection: Anchor-Based (HTML Only)

EDGAR uses consistent `<a name="itemX">` anchors for every Item. Use these instead of relying on `Title` element text matching — they are guaranteed to be stable across filings.

```python
SECTION_MAP = {
    "item1":  "Business Overview",
    "item1a": "Risk Factors",
    "item1b": "Unresolved Staff Comments",
    "item2":  "Properties",
    "item3":  "Legal Proceedings",
    "item7":  "MD&A",
    "item7a": "Market Risk",
    "item8":  "Financial Statements",
    "item9a": "Controls and Procedures",
}

def extract_anchor_sections(soup) -> dict:
    """Returns {anchor_id: section_label} for EDGAR Item anchors."""
    sections = {}
    for anchor in soup.find_all("a", attrs={"name": True}):
        key = anchor["name"].lower().replace(" ", "").replace(".", "")
        if key in SECTION_MAP:
            sections[key] = SECTION_MAP[key]
    return sections
```

Track the current anchor as you iterate elements, and stamp it onto each chunk's metadata as `anchor_id`. This is the most important metadata field for filtered retrieval.

---

## Layer 2 — Chunking Strategy: Route by Element Type

> **Key insight from research (Jimeno-Yepes et al., 2024):** Element-type-based chunking outperforms paragraph-level chunking on financial documents by a significant margin. Structure is signal.

### 2a. Text Chunks — Section-Aware Recursive Splitting

For `NarrativeText` elements, use a **parent-child chunking** approach:

- **Parent chunk**: ~1,500–2,000 tokens — used for context window injection
- **Child chunk**: ~256–512 tokens — used for embedding & retrieval

Why? Small chunks embed with higher precision. But at retrieval time you return the _parent_ for more context. This avoids the "retrieved fragment is too short to be useful" problem.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=512,  chunk_overlap=100)

# Store child for retrieval, parent for generation context
```

**Grouping by section:** In HTML, use the `anchor_id` detected from `<a name="...">` tags rather than raw `Title` text. This is more reliable and gives you a stable key for Qdrant filtering.

```python
current_section = "Unknown"
current_anchor  = None

for element in elements:
    if element.type == "Title":
        # Try to match against known anchor labels
        current_section = element.text
    # anchor_id updated separately from the soup traversal above

    elif element.type == "NarrativeText":
        metadata["section"]   = current_section
        metadata["anchor_id"] = current_anchor   # ← stable EDGAR key
```

**What to skip:** Discard `Header`, `Footer`, elements under 50 characters, and HTML-specific boilerplate: table-of-contents entries (short `Title` elements that are just "Item 1A."), navigation `<div>` blocks, and legal signature pages.

```python
BOILERPLATE_PATTERNS = [
    r"pursuant to the requirements of the securities exchange act",
    r"incorporated herein by reference",
    r"see exhibit index",
    r"^table of contents$",
    r"^item \d+[a-z]?\.$",   # TOC stub entries
]

def should_skip(element) -> bool:
    text = element.text.strip().lower()
    if len(text) < 50:
        return True
    if re.match(r"^\$?[\d,\.]+[bmk]?$", text):  # XBRL numeric artifact
        return True
    return any(re.search(p, text, re.I) for p in BOILERPLATE_PATTERNS)
```

### 2b. Table Chunks — `pd.read_html()` + LLM Summary + Raw Preservation

Tables are the hardest problem. A balance sheet embedded as raw text embeds poorly — the model sees "Assets 15,234 Liabilities 8,891 Equity 6,343" with no structural meaning.

**For HTML filings, use `pd.read_html()` instead of relying solely on `unstructured`'s Table elements.** HTML tables have explicit DOM structure; pandas reads them with full column/row semantics, including merged cells and multi-level headers that `unstructured` can miss.

```python
import pandas as pd

def extract_tables_from_section(html_fragment: str) -> list:
    """Extract all tables from a section's HTML as DataFrames."""
    try:
        return pd.read_html(html_fragment, flavor="lxml")
    except ValueError:
        return []  # no tables found

def df_to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)
```

**Strategy: dual representation**

```
HTML Table element
    ↓
    ├── pd.read_html() → DataFrame → markdown
    │       ↓
    │   LLM Summary → "Apple's 2023 balance sheet shows total assets of $352B,
    │                   total liabilities of $290B, and shareholders' equity of $62B.
    │                   Key items include cash of $61B and long-term debt of $98B."
    │                   → This is what gets EMBEDDED
    │
    └── Raw table markdown (from df.to_markdown())
        → Preserved in Qdrant payload as `table_markdown`
        → This is what gets INJECTED into the LLM context
```

```python
def summarize_table(df: pd.DataFrame, company: str, year: int,
                    section: str, llm) -> str:
    table_md = df_to_markdown(df)
    prompt = f"""You are analyzing a table from {company}'s {year} 10-K filing.
Section: {section}
Describe this table in 2-3 sentences, including all key financial figures.

Table:
{table_md}

Summary:"""
    return llm.invoke(prompt)
```

**Why this works:** You embed the _natural language summary_ (semantically rich), but inject the _raw table markdown_ into the LLM context (numerically precise). Best of both worlds.

### 2c. Metadata — Attach to Every Chunk

Rich metadata enables **filtered retrieval** in Qdrant. For HTML filings, `anchor_id` is the most powerful filter field — it maps directly to EDGAR's Item structure and is far more precise than a free-text section name.

```python
{
    "text": "...",                          # chunk text (or LLM summary for tables)
    "table_markdown": "...",                # raw table markdown (tables only)
    "company": "Apple Inc.",
    "ticker": "AAPL",
    "cik": "320193",                        # SEC CIK — stable company identifier
    "year": 2023,
    "quarter": None,                        # for 10-Q filings
    "filing_type": "10-K",
    "accession_number": "0000320193-23-000106",  # EDGAR unique filing ID
    "section": "Item 1A. Risk Factors",     # human-readable label
    "anchor_id": "item1a",                  # EDGAR anchor — use for Qdrant filtering
    "element_type": "NarrativeText",        # or "Table"
    "chunk_index": 4,
    "parent_chunk_id": "abc123",
    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/.../aapl-20230930.htm",
    "htm_filename": "aapl-20230930.htm"
}
```

> `accession_number` + `anchor_id` together reconstruct a direct link to the exact Item section in the SEC EDGAR viewer — use this for citations in the Answer node.

---

## Layer 3 — Embedding: Finance-Aware Models

Standard `all-MiniLM-L6-v2` is fine for general text but underperforms on financial jargon ("liquidity covenant", "EBITDA margin", "material weakness").

### Recommended Models (in order of preference for a demo)

| Model                                     | Dims | Notes                                              |
| ----------------------------------------- | ---- | -------------------------------------------------- |
| `BAAI/bge-base-en-v1.5`                   | 768  | Strong MTEB scores, good finance performance, free |
| `intfloat/e5-base-v2`                     | 768  | Solid all-rounder, handles financial text well     |
| `text-embedding-3-large` (OpenAI)         | 3072 | Best quality, costs money, good for demo polish    |
| `FinLang/finance-embeddings-investopedia` | 768  | Finance-specific, smaller community, worth testing |

**For this demo:** Use `BAAI/bge-base-en-v1.5` — free, strong, and widely supported.

```python
from fastembed import TextEmbedding, SparseTextEmbedding

dense_model  = TextEmbedding("BAAI/bge-base-en-v1.5")
sparse_model = SparseTextEmbedding("Qdrant/bm25")   # or "prithivida/Splade_PP_en_v1"
```

---

## Layer 4 — Qdrant Schema: Multi-Vector Collection

Store **both dense and sparse vectors** in a single collection. Qdrant natively supports this.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, Modifier
)

client = QdrantClient("localhost", port=6333)

client.create_collection(
    collection_name="sec_filings",
    vectors_config={
        "dense": VectorParams(
            size=768,                    # match your embedding model
            distance=Distance.COSINE
        )
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            modifier=Modifier.IDF        # enables BM25-style IDF weighting
        )
    }
)
```

**Upsert with both vectors:**

```python
from qdrant_client.models import PointStruct, SparseVector
import uuid

points = []
for chunk in chunks:
    dense_vec  = list(dense_model.embed([chunk["text"]]))[0].tolist()
    sparse_out = list(sparse_model.embed([chunk["text"]]))[0]

    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense":  dense_vec,
            "sparse": SparseVector(
                indices=sparse_out.indices.tolist(),
                values=sparse_out.values.tolist()
            )
        },
        payload=chunk  # full metadata dict including anchor_id, source_url, etc.
    ))

client.upsert(collection_name="sec_filings", points=points)
```

---

## Layer 5 — Retrieval: Hybrid Search with Anchor-Filtered RRF Fusion

At query time, run dense + sparse in parallel, fuse with **Reciprocal Rank Fusion (RRF)**, and apply a hard `anchor_id` filter derived from the query's detected intent.

> **Why RRF?** Hybrid search (dense + BM25) achieves 91% recall@10 vs 78% for dense alone — a 17% gain. For financial documents with specific numeric identifiers and tickers, this gap is even larger.

> **Why `anchor_id` filter?** SEC filings have millions of tokens. Filtering to the correct Item before semantic search narrows the candidate pool by ~90%, dramatically improving both precision and latency.

```python
from qdrant_client.models import (
    Prefetch, FusionQuery, Fusion, Filter, FieldCondition, MatchValue
)

# Maps user query intent to EDGAR anchor — set by Query Analyzer node
INTENT_TO_ANCHOR = {
    "risk_factors":      "item1a",
    "business_overview": "item1",
    "financial_data":    "item8",
    "mda":               "item7",
    "market_risk":       "item7a",
    "controls":          "item9a",
}

def hybrid_search(
    query: str,
    ticker: str = None,
    year: int = None,
    anchor_id: str = None,   # ← EDGAR section anchor from Query Analyzer
    top_k: int = 8
):
    conditions = []
    if ticker:
        conditions.append(FieldCondition(key="ticker",    match=MatchValue(value=ticker)))
    if year:
        conditions.append(FieldCondition(key="year",      match=MatchValue(value=year)))
    if anchor_id:
        conditions.append(FieldCondition(key="anchor_id", match=MatchValue(value=anchor_id)))

    query_filter = Filter(must=conditions) if conditions else None

    dense_q  = list(dense_model.embed([query]))[0].tolist()
    sparse_q = list(sparse_model.embed([query]))[0]

    results = client.query_points(
        collection_name="sec_filings",
        prefetch=[
            Prefetch(
                query=SparseVector(
                    indices=sparse_q.indices.tolist(),
                    values=sparse_q.values.tolist()
                ),
                using="sparse",
                limit=20
            ),
            Prefetch(query=dense_q, using="dense", limit=20)
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=query_filter,
        limit=top_k
    )

    return [r.payload for r in results.points]
```

**On retry (CRAG rewrite):** Drop the `anchor_id` filter to broaden scope. The Query Rewriter node should signal this explicitly in the agent state.

---

## Layer 6 — Advanced Retrieval Additions (Do If Time Allows)

### 6a. Contextual Retrieval (Anthropic technique)

Before embedding each chunk, prepend a short context header generated by an LLM. For HTML filings, include the `anchor_id` label — it's a stable, meaningful prefix.

```python
context_prompt = f"""Filing: {company} {year} 10-K
Section: {section} ({anchor_id})
Chunk {i} of {total}:
{chunk_text}"""
# Embed this enriched version instead of raw chunk text
```

This adds ~15% retrieval accuracy on financial benchmarks by reducing embedding ambiguity.

### 6b. Anchor-Based Pre-Filtering (HTML Advantage)

The Query Analyzer maps intent to `anchor_id` before any vector search runs. This is a hard Qdrant filter — not a similarity score adjustment. It reduces the effective search space to a single Item section, which is usually under 5% of the total collection.

```python
# Query: "What are Apple's risk factors in 2023?"
# → extract: { ticker: "AAPL", year: 2023, anchor_id: "item1a" }
# → Qdrant filter applied before RRF fusion
```

This is the single biggest retrieval improvement available when using HTML over PDF — PDF has no equivalent of stable section anchors.

### 6c. HyDE (Hypothetical Document Embeddings)

Instead of embedding the raw user question, ask an LLM to generate a _hypothetical answer passage_, then embed that. Questions embed differently from answers in vector space; HyDE bridges this gap.

```python
hypothetical = llm.invoke(f"Write a short passage from a 10-K that would answer: {question}")
query_vec = embed(hypothetical)  # embed the hypothetical answer, not the question
```

**Note:** Use HyDE only for qualitative questions ("how does X manage risk?"). Skip it for quantitative ones ("what was revenue in Q3?") — HyDE provides no benefit for precise numerical lookups and adds latency.

### 6d. Parent-Child Retrieval in Practice

Store child chunk IDs in Qdrant payload. After retrieval, look up the parent chunk from a local dict (keyed by `parent_chunk_id`) or a second Qdrant collection. Inject _parent_ content into LLM context for fuller context, while keeping child embeddings for precise matching.

### 6e. EDGAR Deep-Link Citations (HTML Only)

Because every chunk carries `accession_number` and `anchor_id`, the Answer node can reconstruct a direct URL to the exact Item section in the SEC EDGAR filing viewer:

```python
def build_edgar_url(cik: str, accession_number: str, anchor_id: str) -> str:
    acc_clean = accession_number.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/"
    # The primary HTM filename is available in chunk payload as htm_filename
    return f"{base}{{htm_filename}}#{anchor_id}"

# e.g. https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm#item1a
```

This makes citations in the answer verifiable with a single click — a significant demo advantage.

---

## Summary Cheatsheet

```
Raw EDGAR HTM
    → clean_edgar_html()              # strip iXBRL tags, hidden XBRL blocks
    → extract_anchor_sections()       # map <a name="item1a"> → section labels
    → partition_html(cleaned, ...)    # unstructured element extraction
    → element-type routing:
         NarrativeText → recursive split (512 child / 2000 parent) + anchor_id stamp
         Table         → pd.read_html() → LLM summary (embed) + markdown (payload)
         Title         → update section tracker
         Header/Footer → discard
    → metadata tagging (ticker, year, cik, accession_number, anchor_id, source_url, ...)
    → Qdrant upsert: dense (bge-base) + sparse (BM25/IDF)

Query
    → Query Analyzer: extract ticker, year → map intent to anchor_id → HyDE if qualitative
    → Qdrant hybrid search: RRF fusion, filtered by ticker + year + anchor_id
    → Relevance Grader: drop irrelevant chunks
    → Context assembly: inject parent chunks + raw table markdown
    → LLM Answer: cited with section label + EDGAR deep-link URL
```

---

## What NOT to Do

| ❌ Don't                                          | ✅ Do instead                                              |
| ------------------------------------------------- | ---------------------------------------------------------- |
| Use `partition_pdf` on EDGAR filings              | Use `partition_html` on the primary HTM document           |
| Parse raw HTM without stripping XBRL first        | Run `clean_edgar_html()` before any parsing                |
| Detect sections by page number or Title text only | Use `<a name="itemX">` anchors — stable across all filings |
| Use `unstructured` alone for complex tables       | Use `pd.read_html()` for precise DataFrame extraction      |
| Embed raw table text                              | LLM-summarize tables, embed summary, store markdown        |
| Use only dense vector search                      | Hybrid: dense + BM25 with RRF                              |
| Filter only by company/year                       | Add `anchor_id` filter — the highest-precision filter available |
| Ignore `accession_number` in metadata             | Store it — enables verifiable EDGAR deep-link citations    |
| Use `all-MiniLM-L6-v2` for finance                | Use `BAAI/bge-base-en-v1.5` minimum                        |
| Embed user question directly for qualitative queries | Use HyDE — then skip it for numerical queries            |
| Drop `anchor_id` filter on all retries            | Loosen filters progressively: keep anchor on retry 1, drop on retry 2 |
