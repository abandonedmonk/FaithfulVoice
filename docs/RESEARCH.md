# 🔬 FaithfulVoice: Research Plan

**Full title:** FaithfulVoice: Real-Time Faithfulness Verification for Voice-Driven Financial RAG Systems  
**One-liner:** You're measuring how often voice AI lies confidently about financial data, building a layer that catches it in real-time, and publishing the dataset.  
**Lives in:** same folder as the rest of the Voice AI / hackathon docs

---

## The Problem

Voice RAG has a trust asymmetry that text RAG doesn't.

When an LLM returns a wrong answer on a screen, the user can re-read it, pause, cross-reference the source. When it speaks the wrong answer with confident intonation, the user just believes it. In financial contexts — earnings figures, risk disclosures, supply chain language from 10-Ks — a subtly wrong spoken claim is genuinely dangerous.

Everyone in the RAG literature measures retrieval quality (did we fetch the right chunks?) or end-to-end answer quality (is the final answer good?). Nobody has specifically measured the gap between retrieved chunk accuracy and spoken response faithfulness in a voice-first financial system.

That gap is the research question.

**Formally:**
> *Do voice-driven financial RAG systems generate spoken responses containing claims unsupported by their retrieved source documents, at what rate, and can an automated real-time verification layer reduce this without unacceptable latency cost?*

---

## Why This Is Publishable

The combination of:
- Voice-specific trust asymmetry (not studied)
- Financial domain RAG (high stakes, well-defined ground truth)
- Real-time verification with latency measurement (practical contribution)
- Released dataset of faithfulness-labeled examples (community contribution)

...does not exist as a single paper. You are the first to write it.

The closest related work (RAGAS, ARES, SelfCheckGPT) measures faithfulness in text RAG systems. None of them study voice, none are domain-adapted to finance, none measure the latency cost of verification in a real-time pipeline. Your related work section writes itself from that gap.

---

## The Two Contributions

**Contribution 1 — The System**
A faithfulness verification layer that sits between RAG output and voice synthesis. Scores every generated response before it is spoken. Flags unsupported claims. Modifies the response if score falls below threshold. Runs in under 50ms.

**Contribution 2 — The Dataset**
Every query through the system auto-generates a labeled example: query + retrieved chunks + generated answer + faithfulness score + flagged claims. 300–500 examples across 4 companies and 5 query domains. Released publicly on HuggingFace Datasets.

Together: one paper. With the fine-tuned verifier model added: a stronger paper.

---

## System Architecture

```
User speaks query
        │
        ▼
Voice pipeline (Nova Sonic or open-source stack)
        │  text query extracted
        ▼
RAG tool fires
  ├── EDGAR fetch or Bedrock KB query
  └── returns: retrieved_chunks (list of strings)
        │
        ▼
LLM generates answer (Llama 3.1 8B)
  └── returns: answer_text
        │
        ▼
┌─────────────────────────────────────┐
│       FAITHFULNESS VERIFIER         │  ← your contribution
│                                     │
│  1. Split answer into claims        │
│     (sentence boundary detection)   │
│                                     │
│  2. For each claim:                 │
│     DeBERTa NLI vs each chunk       │
│     → entailment score 0.0–1.0      │
│                                     │
│  3. Aggregate faithfulness score    │
│  4. Flag unsupported claims         │
│  5. Total latency: ~40ms            │
└──────────────┬──────────────────────┘
               │
        score ≥ 0.7?
         /          \
       Yes            No
        │              │
   Speak normally    Modify response:
                     "Based on the filing,
                      [answer]. I'd recommend
                      verifying the exact figure."
        │              │
        └──────┬────────┘
               │
        Append to audit.jsonl
```

---

## Models — All Free, All Self-Hostable

### Answer Generator (the voice pipeline LLM)

**Primary: Llama 3.1 8B Instruct** (4-bit quantized, via vLLM or llama.cpp on Modal)

Why 8B and not bigger or GPT-4o: deliberate research choice. Using a mid-tier open model makes the verification layer more necessary and results more generalizable. If you used GPT-4o and got high faithfulness, the finding is just "use a better model." Using 8B means the finding is "verification matters regardless of model capability." State this explicitly in the paper — it's a methodological decision, not a budget limitation.

**Comparison: Mistral 7B Instruct** — run the same query set through this for your generator comparison table.

**Stretch (Modal Day 3): Llama 3.1 70B** — run on a subset (50 queries) to see if a much larger generator changes faithfulness rates. Interesting finding either way.

### Faithfulness Verifier (the research contribution)

**Primary: cross-encoder/nli-deberta-v3-base**
- 86MB, runs on CPU in ~40ms
- Trained specifically for NLI/entailment
- Outperforms GPT-3.5 on entailment benchmarks at 1/1000th the cost
- Free, HuggingFace, no API key

**Why not use the same LLM for verification:** Two reasons worth stating in the paper. First, an LLM asked to verify its own output has documented self-consistency bias — it will say its own claims are grounded even when they aren't. Second, a specialized NLI model is faster, cheaper, and more accurate at this specific task. Using the right tool for the job is itself a contribution.

**Ablation models (all free HuggingFace):**

| Model | Size | Latency | Notes |
|-------|------|---------|-------|
| cross-encoder/nli-deberta-v3-small | 44MB | ~20ms | Faster, weaker |
| cross-encoder/nli-deberta-v3-base | 86MB | ~40ms | Primary |
| cross-encoder/nli-deberta-v3-large | 350MB | ~120ms | Stronger, slower |
| facebook/bart-large-mnli | 400MB | ~90ms | Different architecture |
| MoritzLaurer/DeBERTa-v3-large-zeroshot-v2 | 350MB | ~110ms | Strong zero-shot |

These comparisons are your Table 2 — latency vs accuracy tradeoff across verifier choices.

### Fine-Tuned Verifier (stretch goal, Modal compute)

Take your generated dataset, fine-tune DeBERTa-v3-base specifically for financial faithfulness verification. 20 minutes on A100. Test on held-out 20%. If it outperforms the general NLI model on your test set, you have a novel domain-adapted model to release — *FinFaithVerifier*. This is what moves the paper from workshop to main conference territory.

---

## The Dataset

### Schema — every query auto-generates this

```json
{
  "id": "uuid",
  "timestamp": "2025-03-01T14:32:01Z",
  "query": "What did Nvidia say about supply chain risks?",
  "company": "NVDA",
  "filing_type": "10-K",
  "filing_year": 2024,
  "query_domain": "supply_chain",
  "retrieved_chunks": ["...chunk 1...", "...chunk 2..."],
  "chunk_relevance_scores": [0.87, 0.74],
  "generated_answer": "Nvidia cited concentration risk at TSMC...",
  "faithfulness_score": 0.84,
  "claim_level_scores": [
    {
      "claim": "Nvidia cited concentration risk at TSMC",
      "score": 0.91,
      "grounded": true,
      "supporting_chunk_idx": 0
    },
    {
      "claim": "lead times extending to 52 weeks",
      "score": 0.41,
      "grounded": false,
      "supporting_chunk_idx": null
    }
  ],
  "response_modified": false,
  "latency_ms": {
    "retrieval": 340,
    "generation": 820,
    "verification": 43,
    "total": 1203
  },
  "generator_model": "llama-3.1-8b-instruct",
  "verifier_model": "deberta-v3-base-nli"
}
```

### Query domains to cover

Write queries across these 5 domains for each company. ~10 queries per domain per company = 200 queries across 4 companies.

| Domain | Example query | Why interesting |
|--------|--------------|-----------------|
| Revenue figures | "What was Nvidia's total revenue in FY2024?" | Specific numbers — highest hallucination risk |
| Supply chain | "What supply chain risks did AMD disclose?" | Qualitative — lower hallucination risk, good contrast |
| Risk factors | "What regulatory risks did Intel flag?" | Forward-looking language, ambiguous grounding |
| Guidance | "What did Nvidia say about next quarter?" | Speculative by nature — interesting faithfulness edge case |
| Litigation | "What legal proceedings is Apple involved in?" | Factual but obscure — tests retrieval + faithfulness jointly |

### Target scale

- 300 examples: publishable dataset, sufficient for FinNLP workshop
- 500 examples: strong dataset, sufficient for COLING/EACL
- 100 of these manually annotated by 3 people: required for human eval section

---

## How to Automate Data Generation (Not Voice)

This is the practical solution to generating 200+ examples without speaking into a microphone 200 times.

**The key insight:** decouple the voice layer from the research pipeline entirely for data generation. The voice UI is just an input method. What the RAG + verifier actually needs is text. For bulk generation, bypass Nova Sonic and your voice stack completely — call the pipeline function directly with text.

```python
# production path (voice)
microphone → STT → query_text → pipeline() → verifier() → TTS → speaker

# data generation path (automated, no voice involved)
queries.txt → pipeline() → verifier() → audit.jsonl
```

### The generate_dataset.py script

```python
import asyncio
import json
from pathlib import Path
from pipeline import run_rag_pipeline      # your existing RAG function
from verifier import verify_faithfulness   # your verifier
from audit import append_audit_log         # your audit logger

async def generate_dataset(
    queries_file: str = "queries.txt",
    output_file: str = "audit.jsonl",
    company_filter: str = None
):
    queries = Path(queries_file).read_text().strip().split("\n")
    
    for i, line in enumerate(queries):
        # Format: "NVDA|supply_chain|What did Nvidia say about supply constraints?"
        company, domain, query = line.split("|", 2)
        
        if company_filter and company != company_filter:
            continue
        
        print(f"[{i+1}/{len(queries)}] {company} — {query[:60]}...")
        
        # same pipeline your voice app uses — no STT, no TTS
        chunks, answer = await run_rag_pipeline(query, company=company)
        faith_result = verify_faithfulness(answer, chunks)
        
        record = {
            "query": query,
            "company": company,
            "query_domain": domain,
            "retrieved_chunks": chunks,
            "generated_answer": answer,
            **faith_result,
            "generator_model": "llama-3.1-8b-instruct",
            "verifier_model": "deberta-v3-base-nli"
        }
        
        append_audit_log(record, output_file)
        
        # be nice to your own API/local server
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(generate_dataset())
```

### The queries.txt format

```
NVDA|revenue|What was Nvidia's total revenue in fiscal year 2024?
NVDA|revenue|What was Nvidia's data center segment revenue in Q4 2024?
NVDA|supply_chain|What supply chain risks did Nvidia disclose in their latest 10-K?
NVDA|supply_chain|What did Nvidia say about TSMC manufacturing concentration?
NVDA|risk_factors|What regulatory risks did Nvidia flag regarding China sales?
AMD|revenue|What was AMD's total revenue for fiscal year 2024?
AMD|supply_chain|How did AMD describe their foundry relationships?
...
```

One line per query. Simple to write, simple to parse. Writing this file is actual research work — the quality of your query set determines the quality of your dataset. Spend a full day on it. Aim for:
- Variety in specificity (exact numbers vs qualitative claims)
- Variety in answer verifiability (some questions have clear answers in the filing, some are ambiguous)
- Some trick questions where the correct answer is "the filing doesn't say" — these are your hardest faithfulness test cases

### Running on Modal

```python
# modal_runner.py
import modal

app = modal.App("faithfulvoice-dataset-gen")
image = modal.Image.debian_slim().pip_install(
    "vllm", "sentence-transformers", "llama-index", "boto3"
)

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 8,  # 8 hours
    secrets=[modal.Secret.from_name("aws-credentials")]
)
def run_generation():
    import asyncio
    from generate_dataset import generate_dataset
    asyncio.run(generate_dataset())

@app.local_entrypoint()
def main():
    run_generation.remote()
```

```bash
modal run modal_runner.py
# walks away, comes back to audit.jsonl with 200+ entries
```

---

## Results Section — What You Answer With the Dataset

After running the system for a few weeks and generating a few hundred queries, you have a dataset. You analyze it to answer these four questions:

1. **What is the average faithfulness score across domains?** (supply chain vs revenue vs risk factors) — shows where hallucination is worst
2. **Which claim types are most likely to be unfaithful?** (numbers? predictions? comparative claims?) — fine-grained failure mode analysis
3. **Does faithfulness correlate with chunk relevance score?** — tests whether better retrieval would fix the problem or if generation itself is the bottleneck
4. **What's the latency cost of verification — is it acceptable for real-time voice?** — deployability argument

These four answers are your Results section. Everything else in the paper (system design, dataset, experiments) is setup for these.

---

## The Experiments

**Experiment 1 — Baseline faithfulness rates**
200 queries, Llama 3.1 8B, measure average faithfulness score per domain. Expected finding: numerical claims have lower faithfulness scores than qualitative claims. Interesting because it tells practitioners where to be most careful.

**Experiment 2 — Generator model comparison**
Same 200 queries through Mistral 7B. Does generator choice affect faithfulness rate? Comparison table: Llama 3.1 8B vs Mistral 7B, faithfulness score by domain.

**Experiment 3 — Verifier model ablation**
Run all 5 verifier models on your manually annotated 100 examples. Measure accuracy against human labels + latency. This is your latency/accuracy tradeoff table. Shows DeBERTa-base is the sweet spot.

**Experiment 4 — Latency overhead**
Measure end-to-end pipeline latency with and without verification layer across 50 runs. Show the overhead is ~43ms — acceptable for real-time voice. This is your deployability argument.

**Experiment 5 (stretch) — Fine-tuned verifier**
Fine-tune DeBERTa on 80% of your dataset. Test on held-out 20%. Show domain-adapted model outperforms general NLI model on financial faithfulness. Release as FinFaithVerifier on HuggingFace.

---

## Human Evaluation

Non-negotiable for a main conference paper. Workshop is fine without it but include it if possible.

**What to do:** Take 100 query-answer pairs from your dataset. You, your friend (the AWS one), and one more person independently label each claim as faithful/unfaithful with reference to the retrieved chunks. No discussion until after everyone labels independently.

**What to measure:**
- Inter-annotator agreement: Cohen's kappa (sklearn has this — `cohen_kappa_score`)
- Kappa > 0.6 = substantial agreement = your task is well-defined
- Compare human majority label to verifier output: target >80% agreement

**Why it matters:** It validates that your automated metric measures something real. A reviewer's first question is always "does your automated metric correlate with human judgment?" This answers it.

**Time required:** 2 days across 3 people. Do this in April after the hackathon.

---

## Modal Compute Plan (3 days/month A100)

| Day   | What you run                                                        | Output                     |
| ----- | ------------------------------------------------------------------- | -------------------------- |
| Day 1 | Bulk inference — all 200 queries through Llama 3.1 8B + verifier    | Full audit.jsonl dataset   |
| Day 2 | Mistral 7B comparison run + all verifier ablations                  | Experiment 2 and 3 results |
| Day 3 | Fine-tune DeBERTa on dataset + evaluation sweeps + Llama 70B subset | Experiment 5 results       |

Everything else — writing queries, manual annotation, system building, paper writing — uses zero GPU.

---

## Venue Targets

| Venue | Realistic? | What it needs | Deadline (approx) |
|-------|-----------|---------------|-------------------|
| arXiv | Yes, always | Paper in good shape | Immediately when ready |
| FinNLP @ EMNLP 2025 | Yes — primary target | System + dataset + experiments | ~June 2025 |
| COLING 2026 | Stretch | Add fine-tuned model + human eval | ~Aug 2025 |
| EACL 2026 | Stretch | Strong results + clean writing | ~Oct 2025 |
| EMNLP 2025 main | Ambitious | Everything above + very clean results | ~May 2025 |

**FinNLP workshop is the target.** It is not a consolation prize. It is well-regarded, indexed, cited in the financial NLP community, and your paper is directly in scope. A FinNLP paper from a tier 3 Indian college will be noticed by master's admissions committees. It shows you identified a real problem, built a system, measured results, and shipped.

---

## Timeline

```
Mar 1–16    Hackathon
            Build voice pipeline + 4 financial tools
            Wire faithfulness verifier into pipeline
            Audit log running, generating real data

Mar 17–31   Post-hackathon cleanup
            Write queries.txt (200 questions, full day)
            Run bulk inference on Modal Day 1
            First look at faithfulness score distribution

Apr 1–15    Manual annotation (100 examples, 3 people)
            Run all experiments (Modal Days 2-3)
            Compute inter-annotator agreement

Apr 16–30   Write paper draft
            Post to arXiv

May 1       Check FinNLP @ EMNLP 2025 deadline and submit

Jun–Aug     If fine-tuned model results are strong:
            Expand to full paper, submit to COLING
```

---

## Resume / SOP Line

> "Built FaithfulVoice, a real-time faithfulness verification system for voice-driven financial RAG, generating a dataset of 400 faithfulness-labeled query-answer pairs over SEC 10-K filings. Showed LLM-generated spoken financial responses contain unsupported claims at a rate of X%, reducible to Y% with a 43ms NLI verification layer. Preprint: arxiv.org/abs/[id]. Submitted to FinNLP @ EMNLP 2025."

Four numbers. A finding. A venue. A link. The tier 3 college is not the first thing anyone sees.

---

## When Does the Voice Pipeline Actually Run?

A common point of confusion: the project is voice-based, but most experiments never touch audio. Here is exactly when each path is used.

```
Voice app (production):
  You speak "What did Nvidia say about supply chain?"
  → STT converts to text
  → text hits RAG pipeline
  → LLM generates answer
  → verifier scores it
  → TTS speaks it back

Data generation (research):
              "What did Nvidia say about supply chain?"
  → text hits RAG pipeline        ← identical from here down
  → LLM generates answer          ← identical
  → verifier scores it            ← identical
  → saved to audit.jsonl          ← instead of speaking
```

The STT and TTS are the voice wrapper around a text pipeline. For research purposes you remove the wrapper and feed text directly. The thing you are studying — does the LLM generate faithful answers from retrieved chunks — has nothing to do with audio. Audio is just how the user interacts with the system in production.

**Experiment 4 is the only experiment that needs the full production pipeline.** That is the one place where you need the microphone, STT, TTS, and Nova Sonic all running together, because you are measuring real-time latency of the complete system. 50 runs is sufficient for a stable median with variance — maybe 2 hours of actual voice pipeline usage for the entire paper.

| Experiment | Needs audio pipeline? | Why |
|-----------|----------------------|-----|
| 1 — Baseline faithfulness rates | ❌ No | Text → RAG → LLM → verifier |
| 2 — Generator model comparison | ❌ No | Swap LLM, run same text queries |
| 3 — Verifier ablation | ❌ No | Just verifier scoring, no voice |
| **4 — Latency overhead** | **✅ Yes** | Measuring real-time pipeline |
| 5 — Fine-tuned verifier | ❌ No | Training + eval on text dataset |
| Human eval | ❌ No | Annotators read text, not audio |

The paper's system section describes the full production pipeline because that is the deployment context motivating the research. The experiments study the components that matter for the research question. This is standard practice — not a weakness.

---

## Why "Voice-Driven" in the Title Is Justified

This question is worth answering directly because a reviewer may ask it.

The claim the paper makes is not "voice processing causes hallucination." The claim is narrower: **voice delivery of RAG responses creates a trust asymmetry that makes faithfulness more important in this context than in text RAG.** Those are different claims, and only the second one requires text experiments to prove.

The faithfulness failure happens at the LLM generation step — which is identical whether the query arrived via microphone or keyboard. You are not claiming audio waves cause hallucinations. You are claiming that when hallucinations occur in a voice system, the consequences are worse and harder for users to catch.

That argument is supported by:
- Experiment 4 proving the verifier runs within the latency budget of a real voice system
- The system section showing the full production pipeline exists and is deployed
- The introduction grounding the trust asymmetry claim in existing HCI literature (see below)

**The one paragraph that makes the voice framing bulletproof:**

In your introduction, cite 2–3 papers from HCI and speech research showing users exhibit higher trust in spoken information than written information. This is documented — it is called the **media equation effect** (Reeves & Nass, 1996) and there is substantial follow-up work on voice assistant trust specifically. You do not need to run a user study. You cite existing literature to establish the premise, then say "given this asymmetry, faithfulness failures in voice RAG are more consequential than in text RAG, motivating the need for a real-time verification layer." One paragraph, three citations. The voice framing is now academically grounded, not just asserted.

---

## Published Papers That Use This Same Structure

If you are skeptical that text experiments are sufficient for a voice-motivated paper, these published papers use the exact same pattern. Read their methodology sections.

**FaithDial: A Faithful Benchmark for Information-Seeking Dialogue (Dziri et al., 2022)**
Published in Transactions of the ACL — a top-tier venue. Explicitly about dialogue systems, which includes voice assistants. Every single experiment runs on text transcripts. No audio anywhere. The title contains "Dialogue." Published without issue. This is your closest precedent.

**RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)**
arXiv 2309.15217 — the most cited RAG evaluation paper. Motivated by production RAG systems including voice assistants. Every experiment is text in, text out. Nobody questioned whether it was a "real" RAG paper.

**SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection (Manakul et al., 2023)**
arXiv 2303.08896. Motivated by LLMs deployed in real systems where users trust their outputs — including voice systems. All text experiments. Widely cited, standard methodology.

**Measuring and Mitigating Hallucinations in Large Language Models (Mishra et al., 2024)**
Studies hallucination in conversational systems including voice assistants. All text experiments. Standard framing throughout.

The pattern across all of these: the *motivation* is a deployed system with real users. The *experiments* are on text because that is what enables controlled, reproducible, scalable evaluation. Reviewers understand this distinction — it is the norm, not the exception.

**The one scenario where text experiments would not be sufficient:** if your paper claimed that the audio modality itself degrades faithfulness — that speech recognition errors or audio processing specifically causes higher hallucination rates. That would require audio experiments. That is not your claim. Your claim is about what happens after the query becomes text, and all of that is fully measurable with text inputs.

---

## File Map (what lives where in this folder)

```
voice-ai/
├── README.md                          ← core Voice AI project
├── ROADMAP.md
├── TECH_STACK.md
├── MVP.md
├── ARCHITECTURE.md
├── GLOSSARY.md
├── IMPL_hackathon_nova_sonic.md       ← AWS hackathon pivot
├── TEAM_SPLIT.md                      ← you + your friend's task split
├── RESEARCH.md                        ← this file
└── ironclad/                          ← separate Ironclad Agent project
```
