# FaithfulVoice: Experiment Plan

> **Team size:** 4 people (A, B, C, D)
> **GPU budget:** 30 hrs/month Modal A100 (free tier)
> **Queries available:** 1000 (already written)
> **Target:** 300-500 labeled examples for dataset, 100 human-annotated for validation

---

## Execution Order and Dependencies

```
Phase 0: Build Pipeline (prerequisite for everything)
    │
Phase 1: E1 — Dataset Generation
    │         1000 queries through RAG + verifier → audit.jsonl
    │
    ├──────────────────────────────────────────────┐
    │                                              │
Phase 2A: E5 — Human Annotation            Phase 2B: E2 — Generator Comparison
    │         100 examples, 4 annotators           (Modal, runs in parallel)
    │                                              │
Phase 3: E3 + E4 — NLI Ablation + Approach Comparison
    │         needs human labels from E5
    │
Phase 4: E7 — Fine-Tuned Verifier (knowledge distillation)
    │         needs full dataset + human labels for test set
    │
Phase 5: E6 — Latency Overhead
    │         needs voice pipeline working
    │
Phase 6: E8 — Analysis (correlations, archetypes, errors)
              needs ALL above data
```

**Binding constraint:** Modal GPU hours. Phases 1 and 2B use GPU. Everything else is local CPU or API. Plan your Modal sessions carefully — batch all GPU work into 2-3 sessions.

---

## Source Deduplication

test.txt and RESEARCH.md have overlapping experiments. Here is the unified mapping:

| Unified ID | RESEARCH.md source | test.txt source | What it covers |
|------------|-------------------|-----------------|----------------|
| E1 | Exp1 — Baseline faithfulness | (implicit in dataset gen) | Dataset generation: 1000 queries → audit.jsonl |
| E2 | Exp2 — Generator comparison | (not in test.txt) | Llama 8B vs Mistral 7B vs Qwen 7B vs Llama 70B |
| E3 | Exp3 — Verifier ablation | #4 (NLI model list) | All 7 NLI models on 100 human-annotated examples |
| E4 | (not in RESEARCH.md) | #1 (stronger baseline comparison) | NLI vs GPT-4o vs RAGAS vs SelfCheckGPT vs HHEM |
| E5 | Human Evaluation section | #2 (human eval) | 4 annotators, 100 examples, Fleiss' kappa |
| E6 | Exp4 — Latency overhead | #6 (latency cost) | Voice + text pipeline with/without verification |
| E7 | Exp5 — Fine-tuned verifier | #3 + #5 (fine-tune DeBERTa — same experiment listed twice) | FinFaithVerifier via knowledge distillation |
| E8 | Results section questions | #7 (archetype breakdown) + #8 (error analysis) | Synthesis: correlations, archetypes, errors |

**Duplicates removed:** test.txt #3 and #5 are the same experiment (fine-tune DeBERTa). test.txt #4 is a subset now expanded into E3+E4. test.txt #6 = RESEARCH.md Exp4.

---

## E1 — Dataset Generation

Generate the full dataset by running all 1000 queries through the RAG pipeline + DeBERTa-base verifier. This is the foundation for every other experiment.

**Models used:**

| Model | Role | Platform | GPU needed? |
|-------|------|----------|-------------|
| `meta-llama/Llama-3.1-8B-Instruct` | Answer generation | Modal A100 | Yes |
| `BAAI/bge-base-en-v1.5` | Dense embedding (retrieval) | Local CPU (FastEmbed) | No |
| `Qdrant/bm25` | Sparse embedding (retrieval) | Local CPU (FastEmbed) | No |
| `cross-encoder/nli-deberta-v3-base` | Faithfulness scoring | Local CPU (~40ms/answer) | No |

**Where each model runs:**
- Llama 3.1 8B runs on Modal A100 (generation is the GPU-heavy step)
- FastEmbed dense + sparse run locally on CPU (embedding is fast, no GPU needed)
- DeBERTa-base runs locally on CPU (~40ms per answer)
- Qdrant runs locally in Docker

**Team members needed:** 1 (person A monitors Modal run, handles errors/restarts)

**Time:** ~4-6 hours on Modal (1000 queries × ~2-3s each for generation, plus retrieval + verification locally). The bottleneck is Llama generation on GPU.

**Output:** `data/processed/audit.jsonl` — 1000 entries, each with the full schema (query, retrieved_chunks, chunk_relevance_scores, generated_answer, faithfulness_score, claim_level_scores, latency_ms, company, query_domain, generator_model, verifier_model)

**Important:** Run DeBERTa-base as the verifier for ALL 1000 queries. This gives you a baseline faithfulness distribution across domains, companies, and claim types before any comparison or fine-tuning.

---

## E2 — Generator Model Comparison

Run the same 1000 queries through different LLM generators to measure whether generator choice affects faithfulness rates. This answers: "Is low faithfulness a property of the generator, or does it persist across generators?" If all generators produce similar faithfulness rates, the verification layer is universally needed — a much stronger finding than "Llama 3.1 8B hallucinates a lot."

**Models used (all on Modal A100):**

| Model | Arch | Size | Role | Why include | Hours on Modal |
|-------|------|------|------|-------------|----------------|
| `meta-llama/Llama-3.1-8B-Instruct` | Llama | 8B | Baseline generator | Already run in E1, no re-run | 0 (reuse) |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | Qwen | 14B | Distilled reasoning | DeepSeek R1 reasoning patterns distilled into Qwen arch. Tests whether distilled CoT improves faithfulness | ~8 |
| `microsoft/Phi-4-reasoning-plus` | Phi | 14B | Reasoning ceiling | Beats 70B models on AIME/GPQA at just 14B. If this STILL hallucinates → verification universally needed | ~8 |
| `mistralai/Ministral-3-14B-Reasoning-2512` | Mistral | ~14B | Long-context reasoning | 256K context ingests more chunks. Highest AIME (89.8) in 14B class. Replaces weak Mistral-7B (57.6% halluc rate in RAGTruth) | ~8 |
| `meta-llama/Llama-3.1-70B-Instruct` | Llama | 70B | Scale ceiling (100 queries) | Same arch as 8B but 10x params — does scale alone fix faithfulness? | ~2 |

**No duplicate architectures** — Llama, Qwen, Phi, Mistral each represented once. The 70B Llama shares arch with 8B but at 10x scale, answering a distinct question (scale vs reasoning).

**Where each model runs:** All on Modal A100. This is your heaviest GPU experiment. Plan it for one full Modal session (~26 hours).

**Team members needed:** 1 (person A runs Modal, person D can help with data validation)

**Time:** ~26 hours on Modal A100 (3 × 14B models ≈ ~8 hours each for 1000 queries; Llama 70B = ~2 hours for 100 queries)

**Output:** Separate audit files — `audit_deepseek_qwen14b.jsonl`, `audit_phi4_reasoning.jsonl`, `audit_ministral14b.jsonl`, `audit_llama70b_subset.jsonl`. Each has the same schema as E1 but with different `generator_model` field.

**How to analyze:** For each domain, compute average faithfulness_score across generators. Create a table:

| Domain | Llama 8B | DeepSeek-Qwen 14B | Phi-4-reasoning 14B | Ministral 14B | Llama 70B (subset) |
|--------|----------|--------------------|----------------------|---------------|---------------------|
| Revenue | ? | ? | ? | ? | ? |
| Supply chain | ? | ? | ? | ? | ? |
| Risk factors | ? | ? | ? | ? | ? |
| Guidance | ? | ? | ? | ? | ? |
| Litigation | ? | ? | ? | ? | ? |

If all generators show similar faithfulness patterns (e.g., revenue is always worst), the finding is "verification is needed regardless of generator quality or reasoning capability" — your strongest paper contribution. If the 14B reasoning models are better than 8B but still substantially hallucinate, you get the nuanced finding: "reasoning helps but doesn't solve the problem."

---

## E3 — NLI Model Ablation

Run ALL available NLI/verification models on the same 100 human-annotated examples to measure accuracy vs human labels and latency. This produces your Table 2: latency vs accuracy tradeoff across verifier choices.

**Models used (all run locally on CPU — NO GPU needed):**

1. `cross-encoder/nli-deberta-v3-small` (44MB, ~20ms) — speed baseline. If this performs nearly as well as base, you can recommend it for latency-critical deployments
2. `cross-encoder/nli-deberta-v3-base` (86MB, ~40ms) — your primary verifier. The sweet spot you're arguing for
3. `cross-encoder/nli-deberta-v3-large` (350MB, ~120ms) — accuracy ceiling of the DeBERTa family. Shows what you gain by going bigger
4. `facebook/bart-large-mnli` (400MB, ~90ms) — different architecture (BART encoder-decoder vs DeBERTa encoder-only). Important for showing DeBERTa's advantage isn't just "bigger model = better"
5. `MoritzLaurer/DeBERTa-v3-large-zeroshot-v2` (350MB, ~110ms) — zero-shot NLI, not fine-tuned for classic NLI benchmarks. Tests whether general zero-shot reasoning helps or hurts for faithfulness. If this performs worse than fine-tuned DeBERTa, it proves task-specific training matters
6. `ynie/roberta-large-snli_mnli_fever_anli_1_2_3` (~350MB, ~100ms) — RoBERTa architecture trained on 4 NLI datasets (SNLI + MNLI + FEVER + ANLI). Different inductive biases than DeBERTa. If DeBERTa outperforms this despite RoBERTa seeing more NLI data, it suggests DeBERTa's architecture is specifically well-suited for entailment scoring
7. `vectara/hallucination_evaluation_model` (HHEM-2.1-Open, ~600MB, ~1500ms) — purpose-built for hallucination detection. T5-based, trained on RAGTruth benchmark. This is your most direct competitor. If DeBERTa-base matches or beats this at 1/40th the latency, that's a very strong finding

**Where each model runs:** All local CPU. No GPU, no API, no cost. Each model takes ~5-10 minutes to run on 100 examples.

**Team members needed:** 4 (perfectly parallelizable — split the 7 models across 4 people)
- Person A: DeBERTa-small + DeBERTa-base
- Person B: DeBERTa-large + DeBERTa-zeroshot
- Person C: BART-large-MNLI + RoBERTa-large-NLI
- Person D: HHEM-2.1-Open

**Time:** ~1-2 hours total (each person spends 20-30 minutes running their models + 30 minutes formatting results)

**Output:** For each model: list of (claim, human_label, model_score, latency_ms) for all claims in the 100 annotated examples. Combined into a single comparison table:

| Model | Accuracy vs Human | F1 | Latency (ms) | Params |
|-------|-------------------|-----|-------------|--------|
| DeBERTa-small | ? | ? | ~20 | 22M |
| DeBERTa-base | ? | ? | ~40 | 86M |
| DeBERTa-large | ? | ? | ~120 | 304M |
| BART-large-MNLI | ? | ? | ~90 | 400M |
| DeBERTa-zeroshot | ? | ? | ~110 | 304M |
| RoBERTa-large-NLI | ? | ? | ~100 | 355M |
| HHEM-2.1-Open | ? | ? | ~1500 | 250M |

The expected finding: DeBERTa-base is the sweet spot — within 2-3% of the best model at 3-30x less latency. HHEM is interesting but too slow for real-time.

---

## E4 — Approach-Level Comparison

Compare your NLI-based verification approach against fundamentally different approaches to faithfulness evaluation. This is where you prove that NLI is the right tool for the job — not just that DeBERTa is a good NLI model, but that the NLI framing itself outperforms alternative methods.

**Methods compared:**

| # | Approach | Platform | Cost | Latency | Why include |
|---|----------|----------|------|---------|-------------|
| 1 | **DeBERTa NLI** (your approach) | Local CPU | Free | ~40ms | Your contribution. Already run in E3 |
| 2 | **HHEM-2.1-Open** | Local CPU | Free | ~1500ms | Purpose-built hallucination detector. Most direct competitor. Already run in E3 |
| 3 | **LLM-as-Judge (GPT-4o-mini)** | Fireworks API | ~$0.03 for 100 examples | ~800ms | What most practitioners would try first. If 40ms DeBERTa matches this, cost/latency advantage speaks for itself |
| 4 | **RAGAS Faithfulness** | Local + LLM API | ~$1-2 for 100 evals | ~3-5s | Most cited RAG eval framework. Standard baseline. Uses LLM internally for claim splitting + verification |
| 5 | **SelfCheckGPT** | Groq API or Modal | ~$3-5 for 100 queries | ~10-30s | Most rigorous approach (consistency-based). Requires 3-5 generations per query. Clearly impractical for real-time but important comparison point |

**Team members needed:** 2
- Person C: runs RAGAS + SelfCheckGPT (API work, more complex setup)
- Person D: runs GPT-4o-mini judge (simple API calls)

**Time:** ~3-4 hours (mostly API call time + formatting results)

**Output:** Comparison table:

| Approach | Accuracy vs Human | F1 | Latency | Cost/1K queries | Real-time viable? |
|----------|-------------------|-----|---------|-----------------|-------------------|
| DeBERTa NLI | ? | ? | 40ms | $0 | Yes |
| HHEM-2.1-Open | ? | ? | 1500ms | $0 | Marginal |
| GPT-4o-mini Judge | ? | ? | 800ms | $0.15 | Marginal |
| RAGAS | ? | ? | 3-5s | $1-2 | No |
| SelfCheckGPT | ? | ? | 10-30s | $3-5 | No |

This is your money table. It proves the entire paper's thesis: NLI-based verification is the only approach that is accurate AND fast enough AND free.

---

## E5 — Human Annotation

The most critical experiment for paper credibility. Without human labels, you cannot validate that your automated faithfulness scores measure something real. This is what separates "I built a system" from "I validated a system."

**What to annotate:** Take 100 query-answer pairs from the generated dataset (E1). Each annotator independently labels every claim in every answer as faithful or unfaithful. You also compare your labels against DeBERTa's automated scores from E1 — this is the validation step.

**How to select the 100 examples:**
- 20 per company × 5 companies = 100, OR
- 20 per domain × 5 domains = 100
- Pick the domain-based split — it gives you equal coverage across your research question's primary axis

**Annotation protocol for each example:**

Each example comes with both the retrieved chunks AND DeBERTa's claim-level scores from E1. You see DeBERTa's scores **after** you label (to avoid bias), or you label blind and compare afterward.

```
You are given:
1. QUERY: "What did Nvidia say about supply chain risks?"
2. RETRIEVED CHUNKS: [chunk 1 text, chunk 2 text, chunk 3 text]
3. GENERATED ANSWER: "Nvidia cited concentration risk at TSMC and
   mentioned lead times extending to 52 weeks."
4. DEBERTA SCORES (from E1 audit.jsonl):        ← machine's opinion
   - Claim "concentration risk at TSMC":  0.91 (faithful)
   - Claim "lead times to 52 weeks":       0.41 (unfaithful)

Your task:
For EACH sentence/claim in the answer, mark:
- FAITHFUL: the claim is directly supported by the retrieved chunks
- UNFAITHFUL: the claim contradicts or is not supported by the chunks
- UNCLEAR: the claim is partially supported or ambiguous

Claim 1: "Nvidia cited concentration risk at TSMC"
→ Your label: FAITHFUL (chunk 1 explicitly states this)
→ DeBERTa: 0.91 → Agreement ✓

Claim 2: "lead times extending to 52 weeks"
→ Your label: UNFAITHFUL (no chunk mentions 52 weeks)
→ DeBERTa: 0.41 → Agreement ✓

Claim 3: "Gaming revenue grew 15%"
→ Your label: UNFAITHFUL (chunk says 8%)
→ DeBERTa: 0.72 → Disagreement ✗ (DeBERTa was wrong)
```

**Annotation rules:**
1. Each claim is one sentence or one factual assertion (a sentence may contain multiple claims)
2. Judge ONLY against retrieved chunks — do NOT use outside knowledge. A claim can be factually true in the real world but UNFAITHFUL if the chunks don't support it
3. Numbers are claims. "Revenue was $27B" is a claim. If the chunk says "$26.97B" and the answer says "$27B", mark as FAITHFUL (reasonable rounding). If the chunk says "$27B" and answer says "$32B", mark as UNFAITHFUL
4. Comparative claims are claims. "Revenue increased compared to last year" is a claim — check if chunks support it
5. Predictive/forward-looking statements from the filing count as faithful IF the chunks contain them. The question is not "is this prediction accurate" but "did the filing actually say this"
6. "The filing doesn't say" is a FAITHFUL answer if the query asks something not in the chunks
7. Annotate independently — no discussion until ALL 4 have finished all 100 examples

**Annotation template (spreadsheet or simple form):**

| example_id | claim_text | your_label (FAITHFUL/UNFAITHFUL/UNCLEAR) | confidence (1-5) | deberta_score (from E1) | deberta_correct? (Y/N) |
|------------|-----------|------------------------------------------|-------------------|-------------------------|------------------------|

The `deberta_correct?` column is where you explicitly mark whether DeBERTa agreed with your label. This makes the accuracy computation trivial later — just count Y's and N's.

Each person produces one filled template with ~200-400 rows (100 examples × 2-4 claims each).

**Team members needed:** ALL 4

**Time:** ~2-3 days total (each person spends ~4-6 hours on their 100 annotations). Do NOT rush this — annotation quality determines the quality of E3, E4, and E7.

**After annotation:**
1. Compute **Fleiss' kappa** for 4 annotators (`statsmodels.stats.inter_rater.fleiss_kappa` or manual calculation). Target: kappa > 0.6 = substantial agreement
2. Derive **majority label** for each claim (3+ of 4 agree = gold label). If 2-2 split, mark as UNCLEAR and exclude from accuracy calculations
3. These majority labels are your ground truth for E3 and E4
4. **Compute DeBERTa accuracy** — what % of claims did DeBERTa agree with the human majority label? Target: >80% agreement. If DeBERTa agrees 80-90% → it's validated, can be trusted on the other 900 examples. If only 60% → unreliable, you need the fine-tuned version (E7)

**Quick Fleiss' kappa example:**
```python
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
# labels: array of shape (n_claims, n_annotators)
# e.g., [[FAITHFUL, FAITHFUL, UNFAITHFUL, FAITHFUL], ...]
# After converting to 0/1 and aggregating:
# fleiss_kappa(aggregated_table)
```

---

## E6 — Latency Overhead Measurement

Measure whether the verification layer adds unacceptable latency to the real-time voice pipeline. This is your deployability argument.

**Two measurement conditions:**

1. **Text-only pipeline** (no audio): query → retrieval → generation → (with/without verification)
   - 50 runs with verification
   - 50 runs without verification
   - Measure: retrieval_ms, generation_ms, verification_ms, total_ms

2. **Full voice pipeline**: microphone → STT → query → retrieval → generation → verification → TTS → speaker
   - 50 runs with verification
   - 50 runs without verification
   - Measure: stt_ms, retrieval_ms, generation_ms, verification_ms, tts_ms, total_ms

**Models used:**
- Moonshine STT (local CPU or Modal)
- Kokoro TTS (local CPU or Modal)
- Llama 3.1 8B (Modal A100 for generation)
- FastEmbed (local CPU for retrieval)
- DeBERTa-base (local CPU for verification)

**Team members needed:** 2 (A + B run voice pipeline, C + D run text-only)

**Time:** ~2-3 hours (50 runs × ~10-20s each for voice; 50 runs × ~2-3s each for text)

**Output:** Latency distribution table:

| Component | Mean (ms) | Median (ms) | P95 (ms) |
|-----------|-----------|-------------|----------|
| STT | ? | ? | ? |
| Retrieval | ? | ? | ? |
| Generation | ? | ? | ? |
| **Verification** | **?** | **?** | **?** |
| TTS | ? | ? | ? |
| **Total (with verification)** | **?** | **?** | **?** |
| **Total (without verification)** | **?** | **?** | **?** |
| **Overhead** | **?** | **?** | **?** |

Expected finding: verification adds ~40ms overhead on top of ~1-2s total pipeline. That's 2-4% overhead — clearly acceptable for real-time voice.

---

## E7 — Fine-Tuned Verifier (FinFaithVerifier)

Fine-tune DeBERTa-v3-base specifically for financial faithfulness verification. This is your novel domain-adapted model — the contribution that moves the paper from workshop to main conference.

### Method: Knowledge Distillation + Human Validation

Do NOT fine-tune on only 100 human-annotated examples — that's far too little and will overfit. Instead, use a **hybrid approach**:

**Step 1 — Knowledge Distillation (automatic labeling):**
Use a strong LLM (GPT-4o-mini via Fireworks API, or Claude 3.5 Haiku via Anthropic API) to annotate ALL 1000 examples at the claim level. The AI teacher annotates **each factual claim** in each generated answer as `supported` (FAITHFUL) or `unsupported` (UNFAITHFUL), with evidence quoted from the retrieved chunks. These AI labels become the training signal for the DeBERTa student.

**What the AI annotates per example (1000 total):**
- Splits each generated answer into individual claims
- For each claim, marks: `supported: true/false`
- For each claim, provides: `evidence: "exact quote from retrieved chunks"` or `null` if no evidence exists
- Output is a JSON array of claim-level annotations per example

This is NOT the same as human annotation in E5. E5 uses 4 humans on 100 examples for validation. E7 uses 1 AI on all 1000 examples as training data. The AI is the teacher; its labels train the student. Accuracy of the AI teacher is measured in E5 (DeBERTa agreement with humans).

Prompt for the teacher LLM:
```
You are a financial document fact-checker. Given retrieved context from SEC filings
and a generated answer, identify each factual claim in the answer and determine
whether it is supported by the context.

Context: {retrieved_chunks}
Answer: {generated_answer}

Output JSON array:
[{"claim": "...", "supported": true/false, "evidence": "quote from context or null"}]
```

Run this on all 1000 examples. Cost: ~$5-10 total on Fireworks/OpenAI.

**Step 2 — Train/Test Split:**
- **Training set:** 800 examples — labels come from AI teacher (claim-level `supported: true/false` from Step 1)
- **Validation set:** 100 examples — labels come from AI teacher (for early stopping; still AI-labeled, not human)
- **Test set:** 100 examples — labels come from HUMAN annotators (from E5 majority labels) — this is your gold standard

The test set MUST use human-annotated labels, not AI-annotated labels. If you test on AI labels, you're just measuring whether the student mimics the teacher — not whether either is correct.

**Step 3 — Fine-tuning:**
- Base model: `cross-encoder/nli-deberta-v3-base`
- Task: binary classification (entailment vs contradiction) at claim level
- Input format: `premise = retrieved_chunk, hypothesis = claim`
- Training: 3-5 epochs, learning rate 2e-5, batch size 32
- Hardware: Modal A100 (takes ~20 minutes)
- Early stopping on validation set

**Step 4 — Evaluation against human labels:**
Compare on the 100 human-annotated examples:
- DeBERTa-base (before fine-tuning) vs human labels
- FinFaithVerifier (after fine-tuning) vs human labels
- If FinFaithVerifier outperforms base DeBERTa on human labels → domain adaptation works

**Step 5 — Cross-company generalization test:**
Train on 3 companies (NVDA, AMD, AAPL), test on held-out 4th (INTC). This shows the model learns general financial faithfulness patterns, not just company-specific patterns.

**Step 6 — Threshold sensitivity + ROC curve:**
Sweep verification threshold from 0.3 to 0.9 in 0.05 increments. For each threshold, compute precision, recall, F1 against human labels. Plot ROC curve. This determines your optimal operating point and answers the reviewer question "why threshold 0.7?"

**Models used:**
- `cross-encoder/nli-deberta-v3-base` — base model to fine-tune (Local CPU + Modal for training)
- `GPT-4o-mini` or `Claude 3.5 Haiku` — teacher LLM for knowledge distillation (Fireworks/Anthropic API)
- `meta-llama/Llama-3.1-70B-Instruct` — alternative teacher if you want to stay fully open-source (Modal A100)

**Where each model runs:**
- Teacher LLM: API (Fireworks for GPT-4o-mini, ~$5-10 for 1000 examples)
- Fine-tuning: Modal A100 (~20 minutes)
- Inference after fine-tuning: Local CPU (same ~40ms as base DeBERTa)

**Team members needed:** 2
- Person A: fine-tuning on Modal + cross-company generalization test
- Person B: threshold sensitivity analysis + ROC curves

**Time:** ~1 day (API labeling: 2-3 hours, fine-tuning: 30 minutes on Modal, evaluation: 2-3 hours)

**Output:**
1. FinFaithVerifier model weights (uploaded to HuggingFace)
2. Accuracy comparison: base DeBERTa vs FinFaithVerifier on human labels
3. Cross-company generalization numbers
4. ROC curve + optimal threshold

---

## E8 — Analysis

Not a separate experiment but the synthesis of all data into paper-ready findings. Covers the four Results questions plus additional analysis.

**Analyses to perform:**

1. **Average faithfulness score across domains** — from E1 data, group by query_domain, compute mean/std faithfulness_score. Expected: revenue < guidance < risk_factors < supply_chain < litigation (numerical claims hallucinate more)

2. **Claim type failure analysis** — categorize each unfaithful claim by type: numerical, comparative, predictive, factual-but-not-in-context, contradictory. Which types fail most? This is your error analysis (test.txt #8)

3. **Archetype-level score breakdown** — from E1 data, break faithfulness by (company × domain) combinations. This is test.txt #7. Creates a heatmap:

| Company | Revenue | Supply Chain | Risk Factors | Guidance | Litigation |
|---------|---------|-------------|--------------|----------|------------|
| NVDA | ? | ? | ? | ? | ? |
| AMD | ? | ? | ? | ? | ? |
| INTC | ? | ? | ? | ? | ? |
| AAPL | ? | ? | ? | ? | ? |

4. **Retrieval-faithfulness correlation** — from E1 data, plot faithfulness_score vs mean(chunk_relevance_scores). Compute Pearson r. If r << 0.3, the problem is generation (verification needed). If r > 0.7, the problem is retrieval (better retrieval would help). This is the experiment that proves your system's motivation

5. **Chunk count ablation** — run 50 queries with top-k=3, top-k=5, top-k=10 retrieved chunks. Does retrieving more chunks change faithfulness? Quick experiment, 30 minutes on Modal

6. **Error analysis on worst examples** — take the 20 lowest faithfulness scores, manually inspect what went wrong. Categorize: retrieval failure (wrong chunks), generation failure (right chunks but wrong answer), verification failure (wrong NLI score). This tells practitioners where to focus improvement efforts

**Team members needed:** All 4 (collaborative analysis session)

**Time:** ~2 days

**Output:** All figures and tables for the Results section of the paper

---

## Summary: All Models, All Experiments

### Model-to-Platform Map

Where does every model run? One glance:

| Model | Platform | Cost | Experiments |
|-------|----------|------|-------------|
| **Generation (need GPU)** | | | |
| Llama 3.1 8B Instruct | Modal A100 | Free (30hrs/month) | E1, E2 |
| DeepSeek-R1-Distill-Qwen-14B | Modal A100 | Free | E2 |
| Phi-4-reasoning-plus | Modal A100 | Free | E2 |
| Ministral-3-14B-Reasoning | Modal A100 | Free | E2 |
| Llama 3.1 70B Instruct | Modal A100 | Free (subset only) | E2 |
| **Verification (run locally)** | | | |
| DeBERTa-v3-small | Local CPU | Free, ~20ms | E3 |
| DeBERTa-v3-base | Local CPU | Free, ~40ms | E1, E3, E4, E6 |
| DeBERTa-v3-large | Local CPU | Free, ~120ms | E3 |
| BART-large-MNLI | Local CPU | Free, ~90ms | E3 |
| DeBERTa-zeroshot-v2 | Local CPU | Free, ~110ms | E3 |
| RoBERTa-large-NLI | Local CPU | Free, ~100ms | E3 |
| HHEM-2.1-Open | Local CPU | Free, ~1500ms | E3, E4 |
| **API-based (pay per token)** | | | |
| GPT-4o-mini | Fireworks API | ~$0.15/1M tokens | E4 (judge), E7 (teacher) |
| Claude 3.5 Haiku | Anthropic API | ~$0.25/1M tokens | E7 (alt teacher) |
| Groq-hosted Llama/Mistral | Groq API | ~$0.01/1M tokens | E4 (SelfCheckGPT) |

**Rule of thumb:** Generation = Modal GPU (free, limited hours). NLI = Local CPU (free, unlimited). Baselines = API (cheap, < $20 total).

### Generation Models (Modal A100) — Detail

| Model | Experiments | Hours on Modal |
|-------|------------|----------------|
| Llama 3.1 8B Instruct | E1, E2 | ~6 (E1) + 0 (E2 reuse) |
| DeepSeek-R1-Distill-Qwen-14B | E2 | ~8 |
| Phi-4-reasoning-plus | E2 | ~8 |
| Ministral-3-14B-Reasoning | E2 | ~8 |
| Llama 3.1 70B Instruct | E2 (subset) | ~2 |

Total GPU: ~32 hours (tight on 30hr/month free tier — run 14B models across 2 months, or use quantized 4-bit for 14B models to halve GPU time)

### NLI / Verification Models (Local CPU)

| Model | Size | Latency | Experiments | Why Include |
|-------|------|---------|-------------|-------------|
| DeBERTa-v3-small | 44MB | ~20ms | E3 | Speed baseline |
| DeBERTa-v3-base | 86MB | ~40ms | E1, E3, E4, E6 | Primary verifier |
| DeBERTa-v3-large | 350MB | ~120ms | E3 | DeBERTa family ceiling |
| BART-large-MNLI | 400MB | ~90ms | E3 | Different architecture |
| DeBERTa-zeroshot-v2 | 350MB | ~110ms | E3 | Zero-shot comparison |
| RoBERTa-large-NLI | 350MB | ~100ms | E3 | Architectural diversity |
| HHEM-2.1-Open | 600MB | ~1500ms | E3, E4 | Purpose-built hallucination detector |

All run on local CPU. No cost. Each takes 5-10 minutes per 100 examples.

### API Models

| Model | Platform | Experiments | Approx Cost |
|-------|----------|-------------|-------------|
| GPT-4o-mini | Fireworks | E4 (judge), E7 (teacher) | ~$5-10 total |
| Claude 3.5 Haiku | Anthropic | E7 (alternative teacher) | ~$5-10 total |
| Groq-hosted Llama/Mistral | Groq | E4 (SelfCheckGPT generations) | ~$3-5 |

Total API cost: under $20 for all experiments.

---

## Additions & Removals from Original Docs

### Experiments ADDED (not in RESEARCH.md or test.txt)

| Experiment | Why it's critical |
|------------|-------------------|
| **Threshold sensitivity + ROC curve** | You claim threshold 0.7 but never test it. Plot ROC of verifier against human labels across thresholds 0.3–0.9. Without this, a reviewer asks "why 0.7?" and you have no data. (Part of E7) |
| **Retrieval-faithfulness correlation** | Plot faithfulness_score vs chunk_relevance_score. If r << 0.3, the problem is generation → verification matters. If r > 0.7, the problem is retrieval → better retrieval would fix it. This proves your system's motivation. (Part of E8) |
| **Cross-company generalization** | Train FinFaithVerifier on 3 companies, test on held-out 4th. Shows model generalizes to unseen companies, not just memorizing. A single experiment that dramatically strengthens E7. (Part of E7) |
| **Chunk count ablation** | Run 50 queries with top-k=3 vs top-k=5 vs top-k=10. Does retrieving more chunks change faithfulness? Quick: 30 min on Modal. (Part of E8) |

### Models ADDED (not in RESEARCH.md or test.txt)

| Model | Why |
|-------|-----|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | DeepSeek R1 distilled reasoning into 14B Qwen arch. Tests whether distilled CoT improves faithfulness |
| `microsoft/Phi-4-reasoning-plus` | Beats 70B models on AIME/GPQA at 14B. Reasoning ceiling for the ablation |
| `mistralai/Ministral-3-14B-Reasoning-2512` | Highest AIME (89.8) in 14B class, 256K context. Replaces weak Mistral-7B (57.6% halluc rate) |
| `vectara/hallucination_evaluation_model` (HHEM-2.1-Open) | Most direct competitor to your approach. Purpose-built hallucination detector. Beats GPT-4 on RAGTruth benchmark |
| `ynie/roberta-large-snli_mnli_fever_anli_1_2_3` | Architectural diversity — RoBERTa vs DeBERTa vs BART vs T5. Shows DeBERTa is specifically good, not just "big models work" |

### Experiments / Models DOWNGRADED or REMOVED

| What | Action | Why |
|------|--------|-----|
| ~1000 examples for fine-tuning (test.txt #5) | Keep at 800 train + 100 val + 100 test | RESEARCH.md correctly says 300-500 is sufficient. Quality > quantity |
| SelfCheckGPT full investment | Include in comparison table only | Clearly impractical (3-5 generations/query, 10-30s, $3+/query). Don't invest more than a few hours |
| NLI ablation on all 200 queries | Run on 100 human-annotated only | Without human labels you get scores but no accuracy measurement. Use ground truth |
| `MoritzLaurer/DeBERTa-v3-large-zeroshot-v2` | Drop if team is short on time | Interesting but not essential. Zero-shot NLI is a different task. First model to cut |
| `mistralai/Mistral-7B-Instruct-v0.3` | Replaced by Ministral-3-14B-Reasoning | Mistral-7B had highest hallucination rate in RAGTruth (57.6%). 14B reasoning model is a proper upgrade |
| `Qwen/Qwen2.5-7B-Instruct` | Replaced by DeepSeek-R1-Distill-Qwen-14B | Same arch (Qwen) at 14B with distilled reasoning — strictly better for ablation |

---

## Parallelization Summary

| Phase | A | B | C | D | Duration |
|-------|---|---|---|---|----------|
| 1: Dataset Gen | Modal monitor | Free | Free | Local setup | 6h Modal |
| 2A: Annotation | Annotate 100 | Annotate 100 | Annotate 100 | Annotate 100 | 2-3 days |
| 2B: Generator | Modal (DeepSeek-Qwen + Phi-4 + Ministral) | Free | Free | Free | 26h Modal |
| 3: NLI + Approach | DeBERTa-sm/base | DeBERTa-lg/zeroshot | BART/RoBERTa | HHEM + API judges | 4h |
| 4: Fine-tune | Modal fine-tune | Threshold + ROC | Cross-company test | Correlation analysis | 1 day |
| 5: Latency | Voice runs | Voice runs | Text runs | Text runs | 3h |
| 6: Analysis | All together | All together | All together | All together | 2 days |

**Modal GPU hours used:** ~32 hours across 2 months (tight on 30hr/month — split across months or use 4-bit quantization for 14B models to fit in one month). Leaves small buffer for re-runs.
**Total calendar time:** ~10-14 days working part-time, ~7 days working full-time.

### Execution Timeline with Deliverables

| Phase | Days | Experiments | Who | What Gets Produced |
|-------|------|-------------|-----|--------------------|
| 0 | 3-5 | Build pipeline | All | Working RAG + verifier + audit logger |
| 1 | 1 | E1: Dataset gen | A (Modal), B/C (free), D (setup) | `audit.jsonl` with 1000 entries |
| 2A | 2-3 | E5: Human annotation | B, C, D annotate; A free after Modal | Human labels for 100 examples, Fleiss' kappa |
| 2B | 1 | E2: Generator comparison | A runs Modal (parallel with 2A) | `audit_deepseek_qwen14b.jsonl`, `audit_phi4_reasoning.jsonl`, `audit_ministral14b.jsonl`, `audit_llama70b_subset.jsonl` |
| 3 | 1 | E3 + E4: Verifier ablation + approach comparison | Split 4 ways (see table above) | Verifier comparison tables (Table 2 + money table) |
| 4 | 1 | E7: Fine-tuned verifier | A (Modal), B (ROC), C (generalization), D (correlation) | FinFaithVerifier model + ROC + cross-company numbers |
| 5 | 0.5 | E6: Latency overhead | A/B (voice), C/D (text) | Latency breakdown table |
| 6 | 2 | E8: Analysis | All 4 together | Paper-ready figures and tables |
