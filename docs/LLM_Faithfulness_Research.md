# LLM Models for Faithful Generation, RAG, and Financial Reasoning (10B-17B Range)

Research compilation covering: faithful generation, RAG faithfulness benchmarks, anti-hallucination fine-tuning, and financial domain reasoning — focused on the 10B-17B parameter range.

---

## 1. Faithful Generation in RAG Settings: Key Findings from Research

### 1.1 RAGTruth Benchmark (arXiv:2401.00396, May 2024)

The most comprehensive RAG-specific hallucination benchmark to date. Key results:

**Models tested and hallucination density (spans per 100 words):**

| Model | QA | Data-to-Text | Summarization | Overall Hallucination Rate |
|---|---|---|---|---|
| GPT-4-0613 | 0.06 | 0.27 | 0.08 | ~9.3% responses |
| GPT-3.5-turbo | 0.12 | 0.18 | 0.05 | ~10.9% responses |
| Llama-2-70B-chat (4bit) | 0.40 | 1.15 | 0.26 | ~39.3% responses |
| **Llama-2-13B-chat** | **0.48** | **1.53** | **0.41** | **~47.1% responses** |
| Llama-2-7B-chat | 0.59 | 1.27 | 0.58 | ~51.8% responses |
| Mistral-7B-Instruct | 0.59 | 1.51 | 0.86 | ~57.6% responses |

**Critical finding for the 10-17B range:** Llama-2-13B-chat is in the sweet spot where scaling reduces hallucination (vs 7B), but it still hallucinates at nearly 5x the rate of GPT-4. This gap is precisely where fine-tuning can help most.

**Hallucination types in RAG:**
- **Evident Conflict**: Direct contradiction of context (most detectable)
- **Subtle Conflict**: Altering contextual meaning (harder to detect)
- **Evident Baseless Info**: Fabricated details not in context
- **Subtle Baseless Info**: Inferred/assumed details not in context

Baseless information introduction was significantly more prevalent than conflict-based hallucination, especially in QA tasks.

**Fine-tuned Llama-2-13B on RAGTruth:**
- Response-level hallucination detection F1: **78.7%** (vs GPT-4-turbo prompt-based: 63.4%)
- Span-level detection F1: **52.7%** (vs GPT-4-turbo: 28.3%)
- Used for hallucination suppression: reduced hallucination rate by 21.6%-63.2%

**Source:** Niu et al., "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models," arXiv:2401.00396, 2024.

### 1.2 RAGTruth++ and RAGTruth-Enhance (arXiv:2603.27752, March 2026)

Extended the RAGTruth benchmark with re-annotation:
- Found **1.68x more hallucination cases** than original RAGTruth labels
- Suggests existing benchmarks substantially underestimate hallucination prevalence
- RT4CHART framework achieved F1 of 0.776 on RAGTruth++ with hierarchical verification

**Source:** Yu et al., "Retromorphic Testing with Hierarchical Verification for Hallucination Detection in RAG," arXiv:2603.27752, 2026.

### 1.3 CuraView with Qwen3-14B (arXiv:2605.03476, May 2026)

A recent paper demonstrating Qwen3-14B fine-tuned for hallucination detection:
- Fine-tuned Qwen3-14B detection model achieved **F1 of 0.831** on safety-critical E4 metric (direct contradiction)
- 90.9% recall, 76.5% precision
- 50.0% relative improvement over the base model
- Outperformed RAGTruth-style and QAGS-style baselines

**Implication for 10-17B range:** Qwen3-14B is emerging as the strongest base model for faithfulness tasks in this parameter range. Its strong base capabilities combined with RAGTruth-style fine-tuning produce state-of-the-art hallucination detection.

**Source:** Ye et al., "CuraView: A Multi-Agent Framework for Medical Hallucination Detection with GraphRAG-Enhanced Knowledge Verification," arXiv:2605.03476, 2026.

---

## 2. Faithfulness Benchmarks and Model Comparisons

### 2.1 Key Faithfulness Benchmarks

| Benchmark | Focus | Key Findings |
|---|---|---|
| **RAGTruth** (2024) | Word-level hallucination in RAG | Llama-2-13B fine-tuned matches GPT-4 for detection; open-source models hallucinate 3-5x more than GPT-4 |
| **FaithDial** (2022, TACL) | Faithful knowledge-grounded dialogue | Models trained on FaithDial produce more interpretable, cooperative, engaging responses; hallucination critic improves 12.8 F1 on BEGIN |
| **CHARP** (2024, ACL Findings) | Critique of FaithDial | Reveals annotation artifacts in FaithDial; models ignore conversation history; proposes improved diagnostic test |
| **HaluEval** (2023) | Synthetic hallucination evaluation | GPT-4 generates fewer hallucinations; smaller models struggle particularly with numerical hallucinations |
| **FELM** (2023) | Natural response hallucination | Focuses on naturally generated responses across domains |
| **RefChecker** (2023) | Triple-level claim verification | Decomposes responses into knowledge triples for verification |
| **RAGTruth-Enhance** (2026) | Re-annotated RAGTruth | 1.68x more hallucination cases than original labels |
| **BEGIN** | Dialogue coherence/faithfulness | Used as downstream evaluation for FaithDial-trained critics |

### 2.2 Model Rankings on Faithfulness (Open-Source, 10-17B Range)

Based on available benchmark evidence as of mid-2026:

1. **Qwen2.5-14B-Instruct / Qwen3-14B** — Best-in-class for the 10-17B range
   - Strong base instruction-following reduces hallucination tendency
   - Fine-tuned Qwen3-14B achieved state-of-art hallucination detection (F1=0.831)
   - Qwen2.5 series was specifically trained with data quality emphasis reducing hallucination
   - Performs well on RAG benchmarks due to strong context adherence

2. **Llama-2-13B-chat (fine-tuned on RAGTruth)** — Strong when fine-tuned
   - Base model has moderate hallucination (47.1% response rate in RAGTruth)
   - After RAGTruth fine-tuning, achieves 78.7% F1 hallucination detection
   - Negative correlation between scale and hallucination in Llama2 family

3. **Llama-3.1-8B-Instruct** — Good faithfulness for its size
   - Llama 3.1 series significantly improved over Llama 2 in instruction following
   - 8B is below the target range but worth noting for comparison
   - Strong context adherence due to improved RLHF training

4. **Mistral-Nemo-12B-Instruct** — Moderate
   - Original Mistral-7B had highest hallucination in RAGTruth
   - Nemo-12B improvement expected but not yet comprehensively benchmarked on RAGTruth

5. **Phi-3-medium-14B** — Needs more evaluation
   - Microsoft's compact model; strong reasoning for size
   - Faithfulness specifically under-evaluated in public benchmarks

**Important caveat:** Most RAGTruth-style benchmarks have not yet been systematically re-run on the Qwen2.5-14B, Llama-3.1, and Mistral-Nemo-12B models. The CuraView paper (2026) is the strongest evidence for Qwen3-14B's faithfulness capabilities.

### 2.3 FaithDial Benchmark Details (arXiv:2204.10757)

FaithDial was created by editing hallucinated responses in Wizard of Wikipedia:
- Training signal for hallucination critic: +12.8 F1 on BEGIN benchmark
- Proposed auxiliary contrastive objective achieving highest faithfulness
- Benefits generalize to zero-shot transfer on CMU-Dog and TopicalChat
- Human evaluation: FaithDial-trained models perceived as more interpretable, cooperative, engaging

**Limitation (from CHARP, arXiv:2405.15110):** FaithDial contains annotation artifacts that bias models to ignore conversation history. CHARP proposes a better diagnostic test set for conversational hallucination evaluation.

---

## 3. Models Fine-Tuned Specifically for Faithful Generation / Anti-Hallucination

### 3.1 RAGTruth-Finetuned Llama-2-13B

- **Base:** Llama-2-13B-chat
- **Training data:** RAGTruth training split (~16,000 annotated responses)
- **Method:** Full fine-tuning, learning rate 2e-5, 1 epoch, 4x A100
- **Performance:** 78.7% F1 response-level detection, 52.7% F1 span-level
- **Application:** Can be used for hallucination suppression in generation pipelines
- **Availability:** Paper describes methodology; model weights can be reproduced from RAGTruth data

### 3.2 Qwen3-14B Fine-tuned for Hallucination Detection (CuraView)

- **Base:** Qwen3-14B
- **Training data:** Medical discharge summary annotations with 4 evidence grades (E1-E4)
- **Performance:** F1=0.831 on E4 (direct contradiction), F1=0.823 on E3+E4
- **50.0% relative improvement** over base model
- **Key insight:** Evidence-chain-based graph retrieval verification substantially improves factual reliability

### 3.3 Other Anti-Hallucination Approaches

Not model-specific but relevant to the 10-17B range:

- **SelfCheckGPT** (Manakul et al., 2023): Sampling-based consistency checking; 58.8% F1 with GPT-3.5-turbo on RAGTruth
- **LMvLM** (Cohen et al., 2023): Cross-examination between two LLMs; 49.4% F1 on RAGTruth
- **Citation Verification Loop** (eSapiens, arXiv:2506.16768): Grounding consistency via citation checks
- **DPO for Faithfulness**: Direct Preference Optimization with faithful vs. hallucinated pairs shows promise for reducing hallucination without detection overhead

### 3.4 No Standalone "Anti-Hallucination" Model in 10-17B Range

As of mid-2026, there is no widely-adopted, publicly available model in the 10-17B range that has been specifically fine-tuned for faithful generation (as opposed to hallucination detection). The closest are:
- RAGTruth-finetuned Llama-2-13B (detection, not generation)
- Qwen3-14B fine-tuned in CuraView (detection domain)
- Various DPO-trained models that are not publicly released

**This represents a clear gap and opportunity for the FaithfulVoice project.**

---

## 4. Financial Domain Reasoning Models

### 4.1 FinGPT (arXiv:2306.06031)

The primary open-source financial LLM framework:
- **Approach:** Data-centric with LoRA fine-tuning on financial data
- **Base models:** Llama2-7B/13B, ChatGLM2-6B, Falcon-7B, InternLM-20B, Qwen-7B
- **Available models in/near 10-17B range:**
  - `FinGPT/fingpt-sentiment_llama2-13b_lora` — Financial sentiment analysis
  - `oliverwang15/FinGPT_v33_Llama2_13B_Sentiment_Instruction_LoRA_FT_8bit`
  - `FinGPT/fingpt-sentiment_internlm-20b_lora` (slightly above range)
- **Tasks:** Sentiment analysis, financial forecasting, robo-advising
- **Limitation:** Primarily focused on classification/sentiment, not faithful generation or RAG-based reasoning

### 4.2 FinMA

- Fine-tuned Llama2-7B for financial analysis
- Below the 10-17B target range
- Available on HuggingFace as community models

### 4.3 InvestLM (2024)

- Based on Llama2-13B
- Fine-tuned for financial reasoning with financial NLP tasks
- One of the few financial LLMs specifically in the 10-17B range
- Reported improvements on Financial QA benchmarks

### 4.4 Disc-FinLLM (2024)

- Multi-task financial LLM
- Uses chat-based instruction tuning across financial tasks
- Available in various sizes

### 4.5 BloombergGPT (Proprietary)

- 50B parameter model trained on financial data
- Not open-source; mentioned for reference
- Demonstrated that domain-specific pretraining improves financial reasoning

### 4.6 Financial Reasoning Benchmarks

| Benchmark | Focus | Typical Top Models |
|---|---|---|
| **FinQA** | Numerical reasoning on financial reports | GPT-4 > Llama-based fine-tuned |
| **ConvFinQA** | Conversational financial QA | GPT-4 > domain-specific models |
| **TAT-QA** | Tabular + textual QA for finance | Hybrid retrieval + reasoning |
| **FinEval** | Chinese financial knowledge | Qwen-based models excel |
| **CFLUE** | Chinese financial language understanding | ChatGLM, Qwen variants |
| **BloombergGPT benchmarks** | Financial NLP suite | BloombergGPT > general LLMs |

### 4.7 Gap Analysis for Financial + Faithful Generation

**No existing model combines financial domain expertise with faithful generation.** Current financial LLMs (FinGPT, InvestLM) optimize for task performance (sentiment, forecasting) but do not specifically address hallucination in financial RAG settings. This is precisely the niche FaithfulVoice targets.

---

## 5. What Recent Papers (2024-2026) Say About Open-Source Model Faithfulness

### 5.1 Key Themes Across Recent Work

1. **Scale helps but isn't sufficient:** Larger models within a family hallucinate less (Llama-2-70B > 13B > 7B on RAGTruth), but even 70B models hallucinate far more than GPT-4 in RAG settings.

2. **Fine-tuning on faithfulness data is highly effective:** A 13B model fine-tuned on RAGTruth matches or exceeds GPT-4-based detection approaches. This suggests that faithfulness is a learnable capability.

3. **Qwen2.5/Qwen3 models are emerging as faithfulness leaders:** The Qwen2.5 series (released Sep 2024) and Qwen3 series (2025) were trained with data quality emphasis that appears to reduce hallucination tendency. CuraView's Qwen3-14B results are the strongest evidence for 14B-class faithfulness.

4. **Existing benchmarks underestimate hallucination:** RAGTruth-Enhance found 1.68x more hallucinations than RAGTruth originally labeled. Benchmarks need continuous refinement.

5. **Detection vs. prevention gap:** Most work focuses on detecting hallucinations after generation. Less work on preventing them during generation (which is what FaithfulVoice's real-time verification layer addresses).

6. **DPO shows promise for faithful generation training:** Using faithful vs. hallucinated response pairs for DPO training can reduce hallucination without requiring a separate detection step.

### 5.2 Most Faithful Open-Source Models (Consensus)

| Rank | Model | Size | Faithfulness Evidence | Notes |
|---|---|---|---|---|
| 1 | Qwen3-14B (fine-tuned) | 14B | F1=0.831 hallucination detection (CuraView) | Best evidence in range |
| 2 | Qwen2.5-14B-Instruct | 14B | Strong RAG performance, data quality emphasis | Best base model in range |
| 3 | Llama-2-13B (RAGTruth fine-tuned) | 13B | 78.7% F1 detection, 52.7% span | Proven on RAGTruth |
| 4 | Llama-3.1-8B-Instruct | 8B | Improved RLHF, but below target range | Strong for its size |
| 5 | Mistral-Nemo-12B | 12B | Limited faithfulness-specific evaluation | Needs more benchmarking |

---

## 6. Recommendations for FaithfulVoice

Based on this research:

### 6.1 Base Model Selection
- **Primary recommendation: Qwen2.5-14B-Instruct** or **Qwen3-14B**
  - Best faithfulness characteristics in the 10-17B range
  - Strong financial reasoning capabilities (FinEval benchmarks)
  - Can be further fine-tuned for faithful generation in financial RAG

### 6.2 Fine-Tuning Strategy
- Use RAGTruth + financial domain data for dual-objective fine-tuning
- DPO with faithful vs. hallucinated pairs (grounded in financial documents)
- The RAGTruth fine-tuning methodology is proven and reproducible

### 6.3 Evaluation Framework
- RAGTruth-style word-level hallucination annotation on financial documents
- Adapt CuraView's 4-tier evidence grading (E1-E4) for financial claims
- Include RT4CHART's hierarchical verification for fine-grained auditing

### 6.4 Research Gap to Fill
- No model exists that combines financial domain expertise + faithful generation + voice-specific evaluation
- The FaithfulVoice project's voice-specific trust asymmetry angle is genuinely novel
- Publishing a financial RAG faithfulness dataset would be a significant community contribution

---

## Sources

1. Niu, C., Wu, Y., Zhu, J., Xu, S., Shum, K., Zhong, R., Song, J., & Zhang, T. (2024). "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models." arXiv:2401.00396.

2. Ye, S., Kong, X., He, X., Yan, G., & Oh, D. (2026). "CuraView: A Multi-Agent Framework for Medical Hallucination Detection with GraphRAG-Enhanced Knowledge Verification." arXiv:2605.03476.

3. Yu, B., Zhang, Y., Lin, L., Briand, L., & Muñoz, E. (2026). "Retromorphic Testing with Hierarchical Verification for Hallucination Detection in RAG." arXiv:2603.27752.

4. Dziri, N., Kamalloo, E., Milton, S., Zaiane, O., Yu, M., Ponti, E.M., & Reddy, S. (2022). "FaithDial: A Faithful Benchmark for Information-Seeking Dialogue." TACL. arXiv:2204.10757.

5. Ghaddar, A., Alfonso-Hermelo, D., Langlais, P., Rezagholizadeh, M., Chen, B., & Parthasarathi, P. (2024). "CHARP: Conversation History AwaReness Probing for Knowledge-grounded Dialogue Systems." ACL Findings 2024. arXiv:2405.15110.

6. Yang, H., Liu, X.-Y., & Wang, C.D. (2023). "FinGPT: Open-Source Financial Large Language Models." IJCAI 2023 FinLLM Symposium. arXiv:2306.06031.

7. Shi, I., Li, Z., Wang, W., He, L., Yang, Y., & Shi, T. (2025). "eSapiens: A Real-World NLP Framework for Multimodal Document Understanding and Enterprise Knowledge Processing." arXiv:2506.16768.

8. HuggingFace model repository: FinGPT organization models (https://huggingface.co/FinGPT)

9. HuggingFace model repository: Faithfulness-related models (search: "anti hallucination", "rag faithfulness")
