# FaithfulVoice: Build Guide for Core Modules

> These are the modules that appear in your paper's Method section.
> You should write these yourself. This guide tells you exactly what libraries,
> functions, and patterns to use — but you write the code.

---

## Architecture Overview

```
Query string
    │
    ▼
┌─────────────────┐
│  Query Analyzer  │  src/nodes/analyzer.py
│  extract ticker  │
│  intent→anchor   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query Rewriter  │  src/nodes/rewriter.py
│  voice→text      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Retriever       │  src/retriever.py
│  hybrid RRF      │
│  + anchor filter │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Relevance Grader│  src/nodes/grader.py
│  score chunks    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Answer Generator│  src/nodes/answer.py
│  Llama 3.1 8B    │
│  cited answer    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Verifier        │  src/verifier.py
│  DeBERTa NLI     │
│  claim scores    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Pipeline        │  src/pipeline.py
│  orchestrate all │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audit Logger    │  src/audit.py
│  JSONL per run   │
└─────────────────┘
```

---

## Build Order

| Step | Module | Depends on | Why this order |
|------|--------|-----------|----------------|
| 1 | `embedder.py` | chunker.py | Must embed before you can retrieve |
| 2 | `retriever.py` | embedder.py, ingest.py | Core retrieval — test with manual queries |
| 3 | `verifier.py` | (standalone) | Core contribution — test with claim/chunk pairs |
| 4 | `nodes/analyzer.py` | config.yaml | Extracts ticker + intent from query |
| 5 | `nodes/rewriter.py` | (standalone) | Optional — voice queries need cleanup |
| 6 | `nodes/grader.py` | (standalone) | Score chunk relevance before answer gen |
| 7 | `nodes/answer.py` | retriever, config | LLM generation — needs retrieved chunks |
| 8 | `nodes/retriever_node.py` | retriever.py | Thin LangGraph wrapper |
| 9 | `pipeline.py` | all above | Wire everything together |
| 10 | `graph.py` | pipeline, nodes | LangGraph state machine |
| 11 | `audit.py` | pipeline | Log every run for E1 dataset |

---

## Module 1: `src/embedder.py`

### Purpose
Wrap FastEmbed to produce dense + sparse vectors from text. This is what `ingest.py` calls, and what `retriever.py` will call at query time.

### Libraries
- `fastembed.TextEmbedding` — dense embeddings
- `fastembed.SparseTextEmbedding` — sparse (BM25) embeddings
- `yaml` — read config.yaml for model names

### Key Functions to Write

```python
def load_dense_model(model_name: str = None) -> TextEmbedding
```
- If `model_name` is None, read from `config.yaml` → `models.dense_embedding`
- `TextEmbedding(model_name)` — first call downloads model (~400MB for bge-base)
- Return the model object (it's reusable)

```python
def load_sparse_model(model_name: str = None) -> SparseTextEmbedding
```
- Same pattern, read `config.yaml` → `models.sparse_embedding` if not given
- `SparseTextEmbedding("Qdrant/bm25")`

```python
def embed_dense(texts: list[str], model: TextEmbedding) -> list[list[float]]
```
- `model.embed(texts)` returns a **generator** of numpy arrays
- Convert each: `list(generator)` → list of numpy arrays → `.tolist()` each one
- Return `list[list[float]]`

```python
def embed_sparse(texts: list[str], model: SparseTextEmbedding) -> list[dict]
```
- `model.embed(texts)` returns generator of `SparseEmbedding` objects
- Each has `.indices` (numpy int array) and `.values` (numpy float array)
- Convert: `{"indices": sv.indices.tolist(), "values": sv.values.tolist()}`
- Return `list[dict]`

```python
def embed_query(query: str) -> dict
```
- Convenience: embed a single query string into both dense + sparse
- Load models (lazy — cache them as module-level globals)
- Return `{"dense": [...], "sparse": {"indices": [...], "values": [...]}}`

### Important Notes
- FastEmbed models download on first use. The first `embed()` call will be slow. After that, they're cached in `~/.cache/fastembed/`
- `TextEmbedding.embed()` takes `list[str]`, NOT a single string. For one query, wrap it: `list(model.embed([query]))[0]`
- Batch size doesn't matter much for CPU — FastEmbed handles it internally
- The dense vector dim must match `config.yaml` → `qdrant.dense_dim` (768 for bge-base)

### Testing
```python
model = load_dense_model()
vecs = embed_dense(["hello world", "test query"], model)
assert len(vecs) == 2
assert len(vecs[0]) == 768  # bge-base dim
```

---

## Module 2: `src/retriever.py`

### Purpose
Given a query string, retrieve the most relevant chunks from Qdrant using hybrid search (dense + sparse via RRF fusion), optionally filtered by ticker and anchor_id.

### Libraries
- `qdrant_client.QdrantClient` — connect to Qdrant
- `qdrant_client.models` — `Prefetch`, `FusionQuery`, `Fusion`, `Filter`, `FieldCondition`, `MatchValue`, `SparseVector`
- `src.embedder` — your embedder (for query embedding)
- `yaml` — config

### Key Functions to Write

```python
def get_client() -> QdrantClient
```
- Read `config.yaml` → `qdrant.host` and `qdrant.port`
- Return `QdrantClient(host, port=port)`

```python
def hybrid_search(
    query: str,
    ticker: str | None = None,
    anchor_id: str | None = None,
    top_k: int = 8,
    prefetch_limit: int = 20,
) -> list[dict]
```
This is the main function. Here's the exact flow:

1. **Embed the query** using `src.embedder.embed_query(query)` → get dense + sparse vectors

2. **Build filter** (optional):
   ```python
   conditions = []
   if ticker:
       conditions.append(FieldCondition(key="ticker", match=MatchValue(value=ticker)))
   if anchor_id:
       conditions.append(FieldCondition(key="anchor_id", match=MatchValue(value=anchor_id)))
   query_filter = Filter(must=conditions) if conditions else None
   ```

3. **Call `client.query_points()`**:
   ```python
   results = client.query_points(
       collection_name="sec_filings",       # from config
       prefetch=[
           Prefetch(
               query=SparseVector(
                   indices=sparse_vec["indices"],
                   values=sparse_vec["values"],
               ),
               using="sparse",
               limit=prefetch_limit,
           ),
           Prefetch(
               query=dense_vec,
               using="dense",
               limit=prefetch_limit,
           ),
       ],
       query=FusionQuery(fusion=Fusion.RRF),
       query_filter=query_filter,
       limit=top_k,
   )
   ```

4. **Extract payloads**:
   ```python
   return [r.payload for r in results.points]
   ```

   Each payload is the full chunk dict (text, parent_text, ticker, anchor_id, section, etc.)

### The INTENT_TO_ANCHOR Mapping
Read from `config.yaml` → `intents` section. Or hardcode a dict:
```python
INTENT_MAP = {
    "risk_factors": "item1a",
    "business_overview": "item1",
    "financial_data": "item8",
    "mda": "item7",
    # etc.
}
```
The query analyzer (nodes/analyzer.py) will use this to map query intent → anchor_id.

### Important Notes
- **RRF (Reciprocal Rank Fusion)** merges the dense and sparse result lists by rank. No weight tuning needed — that's the beauty of RRF.
- `prefetch_limit` controls how many candidates each vector space returns before fusion. 20 is a good default. Higher = more recall, slower.
- The `using` parameter in Prefetch must match the vector name in the Qdrant collection: `"dense"` and `"sparse"` (as defined in `ingest.py`).
- `SparseVector` takes `indices` and `values` as plain lists (not numpy arrays).
- If Qdrant is not running, `get_client()` will succeed but `query_points()` will throw `ConnectionRefusedError`. Handle it gracefully.

### Testing
```python
results = hybrid_search("What are Nvidia's risk factors?", ticker="NVDA", anchor_id="item1a")
assert len(results) > 0
assert results[0]["ticker"] == "NVDA"
```

---

## Module 3: `src/verifier.py`

### Purpose
Given a generated answer and the retrieved context chunks, score each factual claim for faithfulness using DeBERTa NLI. This IS the paper's core contribution.

### Libraries
- `sentence_transformers.CrossEncoder` — load DeBERTa as a cross-encoder
- `spacy` or `nltk` — sentence splitting for claim extraction
- `yaml` — config for model name and threshold

### Install
```bash
pip install sentence-transformers spacy
python -m spacy download en_core_web_sm
```

### Key Functions to Write

```python
def load_verifier(model_name: str = None) -> CrossEncoder
```
- Default: `cross-encoder/nli-deberta-v3-base` from config
- `CrossEncoder(model_name)` — downloads ~86MB model on first use
- Return the model object (cache as module-level global)

```python
def split_claims(answer: str) -> list[str]
```
- Use spaCy to split answer into sentences: `nlp(answer)` → iterate `doc.sents`
- Each sentence is a "claim" (simplistic but works for E1)
- Advanced: split compound sentences ("A is X and B is Y" → two claims)
  - For now, sentence-level is fine. Claim-level splitting can be improved later.
- Return `list[str]` of individual claims

```python
def score_claim(claim: str, context: str, model: CrossEncoder) -> float
```
- `model.predict([(context, claim)])` returns numpy array of scores
- For NLI cross-encoders, output is 3 logits: [contradiction, neutral, entailment]
- Faithfulness score = softmax(logits)[2] (entailment probability)
- Or simpler: `model.predict()` returns a single score for some cross-encoders. Check:
  - `nli-deberta-v3-base` returns 3-class logits. Apply `scipy.special.softmax()` → take index 2
  - Some cross-encoders return a single similarity score directly
- Return float between 0.0 (contradiction) and 1.0 (entailment)

```python
def verify_answer(
    answer: str,
    context_chunks: list[dict],
    model: CrossEncoder = None,
) -> dict
```
Main function. Flow:
1. Load model if not given
2. Concatenate all chunk texts into one context string:
   ```python
   context = "\n\n".join(c["text"] for c in context_chunks)
   # Also include parent_text for broader context:
   context += "\n\n" + "\n\n".join(c.get("parent_text", "") for c in context_chunks if c.get("parent_text"))
   ```
3. Split answer into claims: `claims = split_claims(answer)`
4. For each claim, score against context: `score_claim(claim, context, model)`
5. Return:
   ```python
   {
       "claims": [{"claim": "...", "score": 0.91}, ...],
       "faithfulness_score": mean of all claim scores,
       "unfaithful_claims": [c for c in claims if c["score"] < threshold],
   }
   ```

### Important Notes
- **Context assembly matters a lot.** If you only pass the child chunk text as context, DeBERTa may mark claims as unfaithful that are actually supported by the parent chunk. Always include parent_text.
- For Table chunks, also include `table_markdown` in the context — numerical claims need the raw table to verify.
- The threshold (0.7 by default) is configurable. You'll sweep it in E7.
- DeBERTa-base takes ~40ms per claim on CPU. An answer with 4 claims takes ~160ms. That's acceptable for real-time.
- `CrossEncoder.predict()` can take a batch of pairs: `model.predict([(ctx1, claim1), (ctx2, claim2), ...])`. Use this for speed — batch all claims at once instead of looping.

### The NLI Framing (Critical for Paper)
The entire verification is framed as **Natural Language Inference**:
- **Premise** = retrieved context chunks
- **Hypothesis** = each claim in the generated answer
- **Entailment** (high score) = claim is faithful — supported by context
- **Contradiction** (low score) = claim contradicts context
- **Neutral** (mid score) = context doesn't address this claim

This is WHY you use a cross-encoder NLI model instead of a general LLM: NLI models are explicitly trained on this exact task (does A entail B?) with massive datasets (MNLI, SNLI, ANLI, FEVER). A general LLM has no such training.

### Testing
```python
model = load_verifier()
context = "NVIDIA reported revenue of $27.0 billion for fiscal year 2024."
claims = ["NVIDIA reported revenue of $27.0 billion.", "NVIDIA reported revenue of $32.0 billion."]
for claim in claims:
    score = score_claim(claim, context, model)
    print(f"  {claim}: {score:.2f}")
# Expect: first ~0.95+, second ~0.1-0.3
```

---

## Module 4: `src/nodes/analyzer.py`

### Purpose
Extract ticker symbol and map query intent to an `anchor_id` for filtered retrieval.

### Libraries
- `re` — regex for ticker extraction
- `yaml` — read config for INTENT_MAP

### Key Functions to Write

```python
def extract_ticker(query: str) -> str | None
```
- Search query for known ticker symbols: NVDA, AMD, INTC, AAPL, MSFT, GOOGL, META, AMZN, TSLA, JPM
- Also check company names: "Nvidia" → NVDA, "Intel" → INTC, etc.
- Return the ticker string or None

```python
def detect_intent(query: str) -> str | None
```
- Use keyword matching to classify the query intent:
  - "risk", "risk factors" → "risk_factors"
  - "revenue", "sales", "income" → "financial_data"
  - "management discussion", "md&a", "outlook" → "mda"
  - "business", "operations", "overview" → "business_overview"
  - "market risk", "interest rate", "currency" → "market_risk"
  - "controls", "procedures", "internal control" → "controls"
  - "legal", "lawsuit", "litigation" → "legal"
  - "cybersecurity", "cyber", "data breach" → "cybersecurity"
- Return the intent key (matches config.yaml → intents keys)

```python
def analyze_query(query: str) -> dict
```
- Combines both: `{"ticker": "NVDA", "intent": "risk_factors", "anchor_id": "item1a"}`
- Maps intent → anchor_id using config.yaml → intents section

### Testing
```python
result = analyze_query("What are Nvidia's supply chain risks?")
assert result["ticker"] == "NVDA"
assert result["anchor_id"] == "item1a"
```

---

## Module 5: `src/nodes/rewriter.py`

### Purpose
Clean up voice-transcribed queries for better retrieval. Voice queries are messy: "uh what did uh Nvidia say about risk factors".

### Libraries
- `re` — simple cleanup

### Key Functions to Write

```python
def rewrite_query(raw_query: str) -> str
```
- Strip filler words: "uh", "um", "like", "you know"
- Strip leading/trailing whitespace
- Fix capitalization for ticker names (nvidia → NVIDIA)
- Remove duplicate words from STT artifacts
- Return cleaned query

### Notes
- This doesn't need an LLM. Simple regex is faster and sufficient for E1.
- If you want to add LLM rewriting later (for complex queries), make it a separate function `rewrite_query_llm()`.

---

## Module 6: `src/nodes/grader.py`

### Purpose
Score each retrieved chunk for relevance to the query. Drop chunks below threshold before passing to answer generation. This reduces noise in the LLM prompt.

### Libraries
- `sentence_transformers.CrossEncoder` — same as verifier but different model
  - Use `cross-encoder/ms-marco-MiniLM-L-6-v2` for chunk grading (fast, ~5ms/pair)
  - Or reuse the bge-base embedder with cosine similarity (cheaper, no new model)

### Key Functions to Write

```python
def grade_chunks(query: str, chunks: list[dict], threshold: float = 0.3) -> list[dict]
```
Two approaches — pick one:

**Approach A: Cross-encoder (better accuracy)**
- `model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
- `scores = model.predict([(query, c["text"]) for c in chunks])`
- Filter: keep chunks where score > threshold
- Sort by score descending

**Approach B: Cosine similarity (no new model needed)**
- Embed query with `src.embedder.embed_dense([query])`
- Embed each chunk text with same model
- Compute cosine similarity between query vec and each chunk vec
- Filter by similarity threshold

Return filtered + sorted chunks.

### Notes
- Approach A is better but requires downloading another model (~80MB).
- Approach B reuses your existing bge-base model — zero extra download.
- Threshold 0.3 is conservative (keep more). 0.5 is aggressive (keep less). Tune on E1 data.
- This step is optional for E1 — you can skip it and just pass all retrieved chunks to the generator. Add it later if you notice irrelevant chunks causing hallucinations.

---

## Module 7: `src/nodes/answer.py`

### Purpose
Generate a cited answer from retrieved chunks using Llama 3.1 8B.

### Libraries
- `openai.OpenAI` — call Llama via Fireworks API (or Modal vLLM)
- `yaml` — config for model name and API base URL

### Key Functions to Write

```python
def generate_answer(
    query: str,
    chunks: list[dict],
    model: str = None,
    base_url: str = None,
    api_key: str = None,
) -> dict
```

Flow:
1. Read config for defaults if not given
2. Build the prompt:
   ```python
   context_parts = []
   for i, c in enumerate(chunks, 1):
       context_parts.append(f"[Source {i} — {c['section']} ({c['anchor_id']})]\n{c['text']}")
       if c.get("parent_text") and len(c["parent_text"]) > len(c["text"]):
           context_parts.append(f"  Extended context: {c['parent_text'][:500]}")
       if c.get("table_markdown"):
           context_parts.append(f"  Table data:\n{c['table_markdown']}")

   context = "\n\n".join(context_parts)

   system_prompt = """You are a financial research assistant. Answer the user's question
   using ONLY the provided context from SEC filings. Cite your sources like [Source 1].
   If the context does not contain enough information, say so explicitly.
   Do not fabricate numbers or claims not supported by the context."""

   user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
   ```

3. Call the LLM:
   ```python
   client = OpenAI(api_key=api_key, base_url=base_url)
   response = client.chat.completions.create(
       model=model,
       messages=[
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": user_prompt},
       ],
       temperature=0.1,
       max_tokens=512,
   )
   answer = response.choices[0].message.content
   ```

4. Return:
   ```python
   {
       "answer": answer,
       "chunks_used": [c["chunk_id"] for c in chunks],
       "model": model,
       "prompt_tokens": response.usage.prompt_tokens,
       "completion_tokens": response.usage.completion_tokens,
   }
   ```

### Important Notes
- **Temperature 0.1** — you want deterministic answers for faithfulness measurement. Don't go above 0.3.
- **max_tokens 512** — SEC filing answers should be concise. If you get rambling answers, reduce to 256.
- The **system prompt is critical** — it's what prevents the LLM from hallucinating. "Using ONLY the provided context" is the key phrase.
- **Citation format [Source N]** — this lets you trace which chunks the answer claims to be based on. The verifier can then check those specific chunks.
- **Fireworks API** is the easiest way to call Llama 3.1 8B. Set `base_url="https://api.fireworks.ai/inference/v1"` and `api_key` from `FIREWORKS_API_KEY` env var. Model name: `"accounts/fireworks/models/llama-v3p1-8b-instruct"`.
- Alternative: run vLLM on Modal and hit it via OpenAI-compatible API. Same code, different base_url.

---

## Module 8: `src/nodes/retriever_node.py`

### Purpose
Thin wrapper around `src/retriever.py` for LangGraph integration.

### Libraries
- `src.retriever` — the actual retrieval logic
- `src.nodes.analyzer` — to extract ticker/anchor from query

### Key Functions to Write

```python
def retrieve(state: dict) -> dict
```
- Input state: `{"query": "What are Nvidia's risk factors?"}`
- Call `analyze_query(state["query"])` → get ticker + anchor_id
- Call `hybrid_search(state["query"], ticker=ticker, anchor_id=anchor_id)`
- Return: `{"chunks": [...], "ticker": ..., "anchor_id": ...}`

### Notes
- This is literally 3 lines of logic. The "node" pattern is just: take state dict → call functions → return updated state dict.
- Don't over-engineer this. The retrieval logic lives in `src/retriever.py`.

---

## Module 9: `src/pipeline.py`

### Purpose
Orchestrate the full RAG + verification pipeline: query → retrieve → grade → generate → verify → return.

### Libraries
- All the modules above: `embedder`, `retriever`, `verifier`, `nodes.*`
- `yaml` — config
- `time` — latency measurement

### Key Functions to Write

```python
def run_pipeline(
    query: str,
    ticker: str | None = None,
    anchor_id: str | None = None,
    config: dict | None = None,
) -> dict
```

Flow (measure latency at each step):
```python
t0 = time.perf_counter()

# 1. Analyze query (if no explicit ticker/anchor)
if ticker is None or anchor_id is None:
    analysis = analyze_query(query)
    ticker = ticker or analysis.get("ticker")
    anchor_id = anchor_id or analysis.get("anchor_id")
t1 = time.perf_counter()

# 2. Retrieve chunks
chunks = hybrid_search(query, ticker=ticker, anchor_id=anchor_id)
t2 = time.perf_counter()

# 3. Grade chunks (optional)
chunks = grade_chunks(query, chunks)
t3 = time.perf_counter()

# 4. Generate answer
answer_result = generate_answer(query, chunks)
t4 = time.perf_counter()

# 5. Verify faithfulness
verification = verify_answer(answer_result["answer"], chunks)
t5 = time.perf_counter()

return {
    "query": query,
    "ticker": ticker,
    "anchor_id": anchor_id,
    "chunks": chunks,
    "answer": answer_result["answer"],
    "faithfulness_score": verification["faithfulness_score"],
    "claims": verification["claims"],
    "unfaithful_claims": verification["unfaithful_claims"],
    "latency_ms": {
        "analyze": (t1-t0)*1000,
        "retrieve": (t2-t1)*1000,
        "grade": (t3-t2)*1000,
        "generate": (t4-t3)*1000,
        "verify": (t5-t4)*1000,
        "total": (t5-t0)*1000,
    },
}
```

### Important Notes
- This function produces the EXACT output schema needed for E1's `audit.jsonl`. Each entry = one call to `run_pipeline()`.
- The latency breakdown is critical for E6. Measure it.
- Don't add retry logic here. If something fails, let it fail — you want to know.
- The function signature accepts optional `ticker` and `anchor_id` so you can override the analyzer for controlled experiments (e.g., force anchor_id="item1a" to test retrieval quality on a specific section).

---

## Module 10: `src/graph.py`

### Purpose
LangGraph state machine for the pipeline. This is the "production" version of `pipeline.py` — same logic but as a proper graph with state management, retry, and potential for streaming.

### Libraries
- `langgraph.graph.StateGraph` — define the graph
- `langgraph.graph.END` — terminal node
- `typing.TypedDict` — define state schema

### Key Functions to Write

```python
class PipelineState(TypedDict):
    query: str
    ticker: str | None
    anchor_id: str | None
    chunks: list[dict]
    answer: str
    faithfulness_score: float
    claims: list[dict]
    latency_ms: dict
```

```python
def build_graph() -> StateGraph
```
- Create `StateGraph(PipelineState)`
- Add nodes: `"analyze"`, `"retrieve"`, `"grade"`, `"generate"`, `"verify"`
- Each node function takes state dict, returns updated state dict (same pattern as `nodes/*.py`)
- Add edges: analyze → retrieve → grade → generate → verify → END
- Compile: `graph = builder.compile()`
- Return compiled graph

### Important Notes
- **Build this LAST.** Get `pipeline.py` working first, then wrap it in LangGraph.
- LangGraph adds checkpointing (resume from any node if it fails), streaming, and observability. Useful for production, unnecessary for E1.
- The graph approach becomes valuable when you want to add loops (e.g., "if faithfulness < 0.5, regenerate with different chunks") or human-in-the-loop (e.g., "if faithfulness < 0.3, flag for human review").
- For E1, `pipeline.py` (simple function) is sufficient. `graph.py` is for the demo and paper architecture diagram.

---

## Module 11: `src/audit.py`

### Purpose
Log every pipeline run to a JSONL file. This is what produces the E1 dataset.

### Libraries
- `json` — serialize results
- `pathlib.Path` — file handling
- `time` — timestamp

### Key Functions to Write

```python
def log_run(result: dict, audit_path: str | Path = "data/processed/audit.jsonl") -> None
```
- Add `timestamp` field to result dict
- Open JSONL in append mode
- Write one line: `json.dumps(result, ensure_ascii=False) + "\n"`
- That's it. Seriously.

```python
def load_audit(audit_path: str | Path = "data/processed/audit.jsonl") -> list[dict]
```
- Read all lines, parse each as JSON
- Return list of dicts
- Useful for analysis scripts

### Important Notes
- Append mode (`"a"`) so you can resume if E1 crashes mid-run
- Add `uuid` per run so you can deduplicate if you re-run
- The audit schema must match what E3/E4/E5/E7 expect. Critical fields:
  - `query`, `answer`, `chunks`, `faithfulness_score`, `claims`, `ticker`, `anchor_id`, `latency_ms`
  - Plus: `company`, `query_domain` (for E8 breakdown)
- Don't over-engineer this. It's a JSONL append. The value is in the data, not the logging code.

---

## Quick Reference: What Each Module Imports

| Module | Imports from |
|--------|-------------|
| `embedder.py` | `fastembed`, `yaml` |
| `retriever.py` | `qdrant_client`, `qdrant_client.models`, `src.embedder`, `yaml` |
| `verifier.py` | `sentence_transformers.CrossEncoder`, `spacy` or `nltk`, `yaml` |
| `nodes/analyzer.py` | `re`, `yaml` |
| `nodes/rewriter.py` | `re` |
| `nodes/grader.py` | `sentence_transformers.CrossEncoder` or `src.embedder` |
| `nodes/answer.py` | `openai`, `yaml` |
| `nodes/retriever_node.py` | `src.retriever`, `src.nodes.analyzer` |
| `pipeline.py` | All of the above, `time` |
| `graph.py` | `langgraph`, `pipeline` logic |
| `audit.py` | `json`, `pathlib` |

---

## Pip Installs You'll Need

```bash
pip install sentence-transformers spacy
python -m spacy download en_core_web_sm
pip install scipy  # for softmax in verifier
```

Everything else (qdrant-client, fastembed, openai, langgraph, yaml) is already installed.
