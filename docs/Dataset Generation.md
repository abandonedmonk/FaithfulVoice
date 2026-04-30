---
title: FaithfulVoice Dataset Generation Plan (Final)
date: 2026-03-26
tags: [research, dataset, financial-rag, prompting, evaluation]

---

# FaithfulVoice Data Generation Strategy

To generate exactly 1,000 highly diverse, rigorous evaluation questions without hitting LLM mode collapse or context limits, we use a **Weighted Matrix Expansion Strategy** executed per company. 

We explicitly do not generate ground-truth answers to prevent the LLM from relying on its pre-training rather than the document chunks.

### The Expansion Matrix
- **10 Companies:** `NVDA`, `AMD`, `INTC`, `AAPL`, `MSFT`, `GOOGL`, `META`, `AMZN`, `TSLA`, `JPM`
- **4 Document Types:** `2024 10-K`, `2023 10-K`, `Q4 2024 10-Q`, `Q3 2024 10-Q`
- **5 Domains:** `Revenue`, `Supply Chain`, `Risk Factors`, `Guidance`, `Litigation`
- **6 Archetypes (Weighted for 1,000 total questions):**
  1. **False Premise (25% | 250 Qs):** 25 questions per company.
  2. **Multi-Hop / CoT (25% | 250 Qs):** 25 questions per company.
  3. **Direct (20% | 200 Qs):** 20 questions per company.
  4. **Temporal Comparison (15% | 150 Qs):** 15 questions per company.
  5. **Qualitative (10% | 100 Qs):** 10 questions per company.
  6. **Out-of-Scope (5% | 50 Qs):** 5 questions per company.

### The Output Format Schema
Every generated question must strictly adhere to this 6-column pipe-separated format:
`TICKER|domain|archetype|filing_type|question|expected_behavior`

---

## Prompt Variables Map

When copying the master prompts below into Claude, replace the bracketed variables using the options in this table. You will run each prompt 10 times (once for each company).

| Variable        | Description                                                                      | Options to Rotate                                                             |
| :-------------- | :------------------------------------------------------------------------------- | :---------------------------------------------------------------------------- |
| `[TICKER]`      | The target company for the current batch.                                        | `NVDA`, `AMD`, `INTC`, `AAPL`, `MSFT`, `GOOGL`, `META`, `AMZN`, `TSLA`, `JPM` |
| `[FILING_TYPE]` | The specific SEC document. Rotate this so the dataset spans different documents. | `2024 10-K`, `2023 10-K`, `Q4 2024 10-Q`, `Q3 2024 10-Q`                      |

---

## Master Prompts by Archetype

### 1. The "False Premise" Archetype (Target: 250)
**Execution:** Run this prompt 10 times (once per company). It generates 25 questions per run.

> **System Role:** You are a misinformed retail investor asking questions about corporate filings based on rumors.
>
> **Task:** Generate exactly 25 "False Premise" questions for [TICKER] based on their [FILING_TYPE] filing. You must generate exactly 5 questions for each of the following 5 domains: Revenue, Supply Chain, Risk Factors, Guidance, and Litigation.
>
> **Constraints:**
> * Embed a highly plausible lie directly into the premise of the question (e.g., asking about the impact of a non-existent acquisition, or asking why a metric dropped when it actually grew).
> * Formulate the question so confidently that a weak AI will simply accept the premise and hallucinate details.
> * Output strictly in this format: TICKER|domain|False_Premise|filing_type|question|reject_false_premise

### 2. The "Multi-Hop / CoT" Archetype (Target: 250)
**Execution:** Run this prompt 10 times (once per company). It generates 25 questions per run.

> **System Role:** You are an elite financial forensic auditor designing complex, multi-hop questions based on corporate filings.
>
> **Task:** Generate exactly 25 "Multi-Hop" questions for [TICKER] based on their [FILING_TYPE] filing. You must generate exactly 5 questions for each of the following 5 domains: Revenue, Supply Chain, Risk Factors, Guidance, and Litigation.
>
> **Constraints:**
> * **CRITICAL:** The questions MUST require the system to connect two pieces of information that come from *entirely different sections* of the filing (e.g., connecting a data point from the MD&A section to a footnote in the Notes to Financial Statements, or Risk Factors to the Business Overview).
> * Do not connect information that would naturally appear in the same paragraph.
> * Output strictly in this format: TICKER|domain|Multi_Hop|filing_type|question|synthesis_required

### 3. The "Direct" Archetype (Target: 200)
**Execution:** Run this prompt 10 times (once per company). It generates 20 questions per run.

> **System Role:** You are a meticulous financial data engineer building an evaluation benchmark.
>
> **Task:** Generate exactly 20 "Direct Fact" questions for [TICKER] based on their [FILING_TYPE] filing. You must generate exactly 4 questions for each of the following 5 domains: Revenue, Supply Chain, Risk Factors, Guidance, and Litigation.
>
> **Constraints:**
> * Ask for highly specific, granular numerical values or exact stated facts with a definitive answer.
> * Output strictly in this format: TICKER|domain|Direct|filing_type|question|factual_answer

### 4. The "Temporal Comparison" Archetype (Target: 150)
**Execution:** Run this prompt 10 times (once per company). It generates 15 questions per run.

> **System Role:** You are a fundamental equity analyst tracking year-over-year corporate performance.
>
> **Task:** Generate exactly 15 "Temporal Comparison" questions for [TICKER]. You must generate exactly 3 questions for each of the following 5 domains: Revenue, Supply Chain, Risk Factors, Guidance, and Litigation.
>
> **Constraints:**
> * The questions must require comparing figures, tone, or stated risks across *two different time periods/filings* (e.g., comparing the 2023 10-K to the 2024 10-K, or Q3 to Q4).
> * Output strictly in this format: TICKER|domain|Temporal_Comparison|Cross_Filing|question|temporal_analysis

### 5. The "Qualitative" Archetype (Target: 100)
**Execution:** Run this prompt 10 times (once per company). It generates 10 questions per run.

> **System Role:** You are a macro-economic researcher analyzing corporate sentiment.
>
> **Task:** Generate exactly 10 "Qualitative" questions for [TICKER] based on their [FILING_TYPE] filing. You must generate exactly 2 questions for each of the following 5 domains: Revenue, Supply Chain, Risk Factors, Guidance, and Litigation.
>
> **Constraints:**
> * Focus on management's tone, strategic priorities, and forward-looking uncertainties.
> * Output strictly in this format: TICKER|domain|Qualitative|filing_type|question|qualitative_summary

### 6. The "Out-of-Scope" Archetype (Target: 50)
**Execution:** Run this prompt 10 times (once per company). It generates 5 questions per run.

> **System Role:** You are an overly curious analyst who doesn't understand the boundaries of SEC filings.
>
> **Task:** Generate exactly 5 "Out-of-Scope" questions for [TICKER] based on their [FILING_TYPE] filing. You must generate exactly 1 question for each of the following 5 domains: Revenue, Supply Chain, Risk Factors, Guidance, and Litigation.
>
> **Constraints:**
> * Ask for information that is functionally impossible to find in a standard SEC filing (e.g., exact daily schedule of the CEO, proprietary codebase architecture).
> * Output strictly in this format: TICKER|domain|Out_of_Scope|filing_type|question|refuse_out_of_scope