# 📄 SEC Filing Chunking & Retrieval Strategy

> Why plain RAG fails on SEC filings — and exactly what to do instead.

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
| XBRL tags / HTML artifacts                                      | Dirty text confuses embeddings                                        |

**The fix is a multi-layer strategy: parse by structure, chunk by element type, retrieve with hybrid search.**

---

## Layer 1 — Parsing: Use `unstructured`

Don't use `PyPDF2` or raw `BeautifulSoup`. Use the `unstructured` library, which understands document structure and returns typed elements.

```python
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf

# For EDGAR HTML filings (preferred — cleaner than PDF)
elements = partition_html(filename="aapl-10k-2023.htm")

# For PDF
elements = partition_pdf(filename="aapl-10k-2023.pdf", strategy="hi_res")
```

`unstructured` returns a list of typed elements:

| Element Type        | What it is                                     |
| ------------------- | ---------------------------------------------- |
| `Title`             | Section header (e.g., "Item 1A. Risk Factors") |
| `NarrativeText`     | Prose paragraphs                               |
| `Table`             | Detected table with row/col structure          |
| `ListItem`          | Bullet/numbered list items                     |
| `Header` / `Footer` | Page metadata — usually discard                |
| `PageBreak`         | Structural marker                              |

This element-type awareness is the foundation of everything that follows.

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

**Grouping by section:** Before splitting, group consecutive elements under the same `Title` ancestor. This means a chunk always knows it belongs to "Item 1A — Risk Factors" rather than being an orphan paragraph.

```python
# Pseudo-code
current_section = "Unknown"
for element in elements:
    if element.type == "Title":
        current_section = element.text
    elif element.type == "NarrativeText":
        chunk_text = element.text
        metadata["section"] = current_section
```

**What to skip:** Discard `Header`, `Footer`, elements with fewer than 50 characters, and boilerplate (detected via regex patterns like "Pursuant to the requirements of the Securities Exchange Act").

### 2b. Table Chunks — LLM Summary + Raw Preservation

Tables are the hardest problem. A balance sheet embedded as raw text embeds poorly — the model sees "Assets 15,234 Liabilities 8,891 Equity 6,343" with no structural meaning.

**Strategy: dual representation**

```
Table element
    ↓
    ├── LLM Summary → "Apple's 2023 balance sheet shows total assets of $352B,
    │                   total liabilities of $290B, and shareholders' equity of $62B.
    │                   Key items include cash of $61B and long-term debt of $98B."
    │                   → This is what gets EMBEDDED
    │
    └── Raw table markdown → Preserved in Qdrant payload as `table_markdown`
                              → This is what gets INJECTED into the LLM context
```

```python
def summarize_table(table_element, llm):
    table_html = table_element.metadata.text_as_html
    prompt = f"""You are analyzing an SEC filing table.
    Describe this table's content in 2-3 sentences, including key figures.
    Table:
    {table_html}
    Summary:"""
    return llm.invoke(prompt)
```

**Why this works:** You embed the _natural language summary_ (semantically rich), but inject the _raw table_ into the LLM context (numerically precise). Best of both worlds.

### 2c. Metadata — Attach to Every Chunk

Rich metadata enables **filtered retrieval** in Qdrant, which is often more important than semantic similarity for SEC filings.

```python
{
    "text": "...",                      # chunk text (or summary for tables)
    "table_markdown": "...",            # raw table (tables only)
    "company": "Apple Inc.",
    "ticker": "AAPL",
    "year": 2023,
    "quarter": None,                    # for 10-Q
    "filing_type": "10-K",
    "section": "Item 1A. Risk Factors",
    "element_type": "NarrativeText",    # or "Table"
    "chunk_index": 4,                   # position within filing
    "parent_chunk_id": "abc123",        # for parent-child linking
    "page": 34
}
```

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
        payload=chunk  # full metadata dict
    ))

client.upsert(collection_name="sec_filings", points=points)
```

---

## Layer 5 — Retrieval: Hybrid Search with RRF Fusion

At query time, run dense + sparse in parallel and fuse with **Reciprocal Rank Fusion (RRF)**.

> **Why RRF?** Hybrid search (dense + BM25) achieves 91% recall@10 vs 78% for dense alone — a 17% gain with minimal latency cost (~6ms). For financial documents with specific numeric identifiers and tickers, this gap is even larger.

```python
from qdrant_client.models import (
    Prefetch, FusionQuery, Fusion, Filter, FieldCondition, MatchValue
)

def hybrid_search(query: str, company: str = None, year: int = None, top_k: int = 8):
    # Build optional metadata filter
    conditions = []
    if company:
        conditions.append(FieldCondition(key="ticker", match=MatchValue(value=company)))
    if year:
        conditions.append(FieldCondition(key="year", match=MatchValue(value=year)))

    query_filter = Filter(must=conditions) if conditions else None

    # Embed query
    dense_q  = list(dense_model.embed([query]))[0].tolist()
    sparse_q = list(sparse_model.embed([query]))[0]

    # Hybrid search with RRF fusion
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
            Prefetch(
                query=dense_q,
                using="dense",
                limit=20
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=query_filter,
        limit=top_k
    )

    return [r.payload for r in results.points]
```

---

## Layer 6 — Advanced Retrieval Additions (Do If Time Allows)

### 6a. Contextual Retrieval (Anthropic technique)

Before embedding each chunk, prepend a short context header generated by an LLM:

```python
context_prompt = f"""Filing: {company} {year} 10-K, Section: {section}
Chunk {i} of {total}:
{chunk_text}"""
# Embed this enriched version instead of raw chunk text
```

This adds ~15% retrieval accuracy on financial benchmarks by reducing ambiguity about what a chunk is about.

### 6b. Metadata-Filtered Pre-Search

Before running hybrid search, use the **Query Analyzer** node to extract entities (company name, year, section). Then pass as hard Qdrant filters. Filtering narrows the search space before semantic similarity runs — much faster and more precise.

```python
# Query: "What are Apple's risk factors in 2023?"
# → extract: { ticker: "AAPL", year: 2023, section_hint: "Risk Factors" }
# → Filter in Qdrant before semantic search
```

### 6c. HyDE (Hypothetical Document Embeddings)

Instead of embedding the raw user question, ask an LLM to generate a _hypothetical answer passage_, then embed that. Questions embed differently from answers in vector space; HyDE bridges this gap.

```python
hypothetical = llm.invoke(f"Write a short passage from a 10-K that would answer: {question}")
query_vec = embed(hypothetical)  # embed the hypothetical answer, not the question
```

**Note:** Recent benchmarks show HyDE provides limited benefit for precise numerical queries (exact dollar figures). Use it for qualitative questions ("how does X manage risk?") but skip for quantitative ones ("what was revenue in Q3?").

### 6d. Parent-Child Retrieval in Practice

Store child chunk IDs in Qdrant payload. After retrieval, look up parent chunk from a local dict or a second Qdrant collection. Inject _parent_ content into LLM context for fuller context.

---

## Summary Cheatsheet

```
Document → unstructured.partition_html()
         → element-type routing
              NarrativeText → recursive split (512 child / 2000 parent)
              Table         → LLM summary (embed) + raw markdown (payload)
              Title         → section label only
         → metadata tagging (company, year, section, element_type, ...)
         → Qdrant upsert with dense (bge-base) + sparse (BM25/IDF)

Query    → Query Analyzer (extract entities → build filter)
         → HyDE if qualitative question
         → Qdrant hybrid search (RRF fusion, filtered by metadata)
         → Relevance Grader (drop bad chunks)
         → Context assembly (inject parent chunks + raw tables)
         → LLM Answer with citations
```

---

## What NOT to Do

| ❌ Don't                                | ✅ Do instead                               |
| --------------------------------------- | ------------------------------------------- |
| Split by fixed token count              | Split by element type + section boundary    |
| Embed raw table text                    | LLM-summarize tables, embed summary         |
| Use only dense vector search            | Hybrid: dense + BM25 with RRF               |
| Ignore document metadata                | Tag every chunk, use Qdrant payload filters |
| Use `all-MiniLM-L6-v2` for finance      | Use `BAAI/bge-base-en-v1.5` minimum         |
| Chunk across section boundaries         | Group elements under parent `Title` first   |
| Embed user question directly for tables | Use HyDE for qualitative queries            |
