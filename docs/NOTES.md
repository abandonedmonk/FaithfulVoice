Tests:
1. A stronger baseline comparison. Don't just show DeBERTa NLI works. Compare against: GPT-4 as verifier, RAGAS faithfulness metric, SelfCheckGPT, and your NLI approach. Show yours is better or comparable at 1/100th the latency. Now you have a proper comparison table.
2. A human evaluation component. Have 3 people (you, your friend, one more) annotate 100 query-answer pairs for faithfulness. Compute inter-annotator agreement (Cohen's kappa). Compare human judgments to your automated scores. This is what separates "I built a system" from "I validated a system." Main conference papers almost always have human eval. It's maybe 2 days of work and dramatically strengthens the paper.
3. A fine-tuned model. Take your generated dataset, use it to fine-tune a small model specifically for financial faithfulness verification. Show it outperforms the general NLI model on your test set. This is a genuine technical contribution — a domain-adapted financial faithfulness verifier that didn't exist before. This is the thing that makes a main conference reviewer say "okay, this is novel."
4. Primary: cross-encoder/nli-deberta-v3-base from HuggingFace. 86MB, runs on CPU in ~40ms, trained specifically for NLI/entailment. Free, no API, no cost. For your ablation table, also run:
a. cross-encoder/nli-deberta-v3-small — faster, slightly weaker
b. facebook/bart-large-mnli — different architecture, good comparison point
c. MoritzLaurer/DeBERTa-v3-large-zeroshot-v2 — stronger but slower
5. Take your generated dataset (~1000 examples), fine-tune DeBERTa-v3-base on it specifically for financial faithfulness verification. This is your domain-adapted model — FinFaithVerifier or whatever you want to call it. This is what makes the paper novel enough for a main conference rather than just a workshop. Fine-tuning DeBERTa on 300 examples takes maybe 20 minutes on an A100.
6. Latency cost of verification: Measure end-to-end voice latency with and without the verification layer. Show the overhead is acceptable (<50ms) for real-time voice. This answers the practical question of whether the system is deployable.
7. Archetype-level score breakdown
8. Error analysis on worst examples
