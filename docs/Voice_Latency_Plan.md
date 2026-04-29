# 9 — Voice Latency Experiment Plan

> Experiment 4 of FaithfulVoice: 50 end-to-end voice pipeline runs measuring per-component latency with and without the faithfulness verifier, proving the verifier adds <50ms overhead and is deployable in a real-time voice system.

---

## What It Is

A fully open-source, self-hostable, **pure Python** voice pipeline deployed on Modal:

```
Audio → Moonshine STT → RAG+LLM (Llama 3.1 8B) → Verifier (DeBERTa) → Kokoro TTS → Audio
```

50 runs **with** verifier, 50 runs **without**. Measure per-component latency. Show overhead ≈ 40ms — imperceptible in real-time conversation.

---

## Why This Way Only

### Why Pure Python (Not Rust)

Rust adds no value for this experiment. The bottleneck is **model inference** (800ms+ for the LLM), not orchestration glue code. Python's overhead for calling Moonshine, awaiting vLLM, scoring with DeBERTa, and calling Kokoro is <<1ms per component — **0.1% of total pipeline latency**, undetectable in the results.

| Factor | Rust | Python |
|--------|------|--------|
| Bottleneck | Model inference (same in both) | Model inference (same in both) |
| Model integration | Requires subprocess calls to Python, adding IPC overhead | Direct API calls — zero overhead |
| What Rust gives you | Real-time WebSocket serving, VAD barge-in, state machine | None of that needed — 50 sequential pre-recorded clips |
| Modal SDK | None — would need custom HTTP server | Native SDK with decorators |
| Development time | Weeks (current pipeline proves it) | Days |
| Reproducibility for reviewers | Requires Rust toolchain + Python subprocess setup | `pip install` + one command |

The existing Rust pipeline in this repo was built for **live concurrent users with barge-in**. Experiment 4 is **50 sequential pre-recorded audio clips** with no live user, no interruption, no concurrency. The Rust engineering (tokio channels, ONNX VAD, state machine) solves problems this experiment doesn't have.

The paper's claim is that the **verifier** runs within latency budget, not that the glue code is fast. Reviewers care that the numbers are reproducible and the verifier overhead is measured correctly — not what language orchestrates the function calls.

### Why Pipeline Architecture (Not Nova Sonic)

The paper's core contribution is a **faithfulness verification layer inserted between LLM output and TTS input**. This requires a pipeline where text is intercepted before becoming audio.

Nova Sonic is a unified speech-to-speech model — it generates text and audio together internally. You **cannot insert a text-based NLI verifier between its reasoning and its speech output**. The best you could do is post-hoc verification (verify after it's already speaking, then interrupt), which changes the contribution from "real-time pre-TTS prevention" to "after-the-fact detection." That's a different paper.

The pipeline architecture isn't just a preference — it's the **only architecture that matches the paper's claim**.

### Why Open-Source Models (Not Deepgram)

The research plan states "All Free, All Self-Hostable" as a hard principle. Deepgram is proprietary, API-only, and costs money. For a reproducibility-focused paper, reviewers and future researchers need to be able to run the exact same system. Open-source models with permissive licenses (MIT, Apache-2.0) ensure:

1. **Any researcher can reproduce Experiment 4** — download models from HuggingFace, run on Modal or local GPU
2. **No API cost barrier** — no signup, no credit card, no rate limits
3. **Version-locked** — model weights are fixed, unlike cloud APIs that silently update
4. **Citeable** — each model has a published paper or technical report with a BibTeX entry

### Why These Specific Models

#### STT: Moonshine Small Streaming

| Attribute | Value |
|-----------|-------|
| Latency (streaming) | 73–165ms (GPU/CPU) |
| WER | 7.84% |
| Streaming | Native, with encoder/decoder state caching |
| License | MIT |
| VRAM | ~150MB |
| Maintenance | Active (2026, regular releases) |

Why not the alternatives:

| Model | Why Not |
|-------|---------|
| **faster-whisper** | 150–400ms, no native streaming (needs Whisper-Streaming wrapper), 30-second fixed window wastes compute on short utterances. Whisper was designed for offline batch transcription, not real-time. |
| **whisper.cpp** | 200–500ms, C++ binary (harder to integrate into Python pipeline), same 30-second window problem |
| **NVIDIA Parakeet/Nemotron** | Good latency (80–240ms) but NVIDIA Open Model License has vendor-specific terms, and NeMo is a heavy framework dependency (~2GB VRAM) |
| **Moonshine Medium** | Better WER (6.65%) but 107–269ms latency — Small hits the sweet spot for latency |

Moonshine was **designed for streaming from the ground up** — it caches intermediate state between chunks (unlike Whisper which recomputes everything). Variable-length input windows mean no wasted compute. It's the only model that achieves sub-100ms streaming latency with a fully permissive license.

#### TTS: Kokoro-82M

| Attribute | Value |
|-----------|-------|
| Latency (per sentence, GPU) | 30–80ms |
| License | Apache-2.0 (code + weights) |
| VRAM | <500MB |
| Quality | Won TTS Arena competitions |
| Maintenance | Active (2024–2026) |

Why not the alternatives:

| Model | Why Not |
|-------|---------|
| **Piper TTS** | Faster (<20ms) but archived (Oct 2025), GPL dependency on espeak-ng, quality noticeably robotic |
| **Fish Speech S2 Pro** | Best streaming latency (~100ms TTFA on H200) but 4B params (8–10GB VRAM), custom research license with commercial restrictions |
| **Coqui XTTSv2** | Streaming capable (~200ms) but company shut down, non-commercial model license (CC BY-NC-NC 4.0), unmaintained |
| **Parler-TTS** | Apache-2.0 (ideal license) but 1–5s per sentence — way too slow for real-time |
| **StyleTTS2** | ~100–200ms latency but GPL dependency on espeak-ng, no native streaming, low maintenance |
| **Bark** | >3s/sentence, no streaming — not real-time capable |
| **ChatTTS** | Non-commercial model license (CC BY-NC 4.0), AGPL code license, no streaming, ~1.5–3s latency |

Kokoro is the only TTS that simultaneously achieves: sub-100ms latency, Apache-2.0 license (no GPL taint, no non-commercial restriction), active maintenance, competitive quality, and <500MB VRAM.

### Why Modal (Not Ray Serve)

| Factor | Modal | Ray Serve |
|--------|-------|-----------|
| Already planned for Experiments 1–3, 5 | Yes | No |
| Production voice streaming examples | Yes (QuiLLMan, Kyutai STT) | No |
| Cold start control | `min_containers`, `scaledown_window`, Memory Snapshots | Manual replica management |
| Academic credits | $10,000 | $100 |
| Per-second billing | Yes | Per-hour minimums likely |
| WebSocket support | Battle-tested (RFC 6455, production voice chat) | Works but less mature |
| Code for Experiment 4 | ~100–150 lines Python | ~300+ lines + YAML config |

Modal is the right choice because:

1. **Unified platform** — Experiments 1–3, 5 already planned on Modal. One platform, one account, one billing.
2. **Academic credits** — $10,000 covers the entire project. Ray's $100 would be tight.
3. **Production voice examples** — QuiLLMan (voice chat with WebSocket streaming) is literally the pattern we need. Fork it, swap models.
4. **Cold start elimination** — `min_containers=1` ensures zero cold starts during the 50-run measurement window. Clean latency numbers.

---

## How To Do It

### Phase 1: Model Preparation (Local Validation)

1. **Download Moonshine Small** from HuggingFace (`moonshine-ai/moonshine-small`). Verify it transcribes 50 test audio clips correctly locally. Measure baseline latency.

2. **Download Kokoro-82M** from HuggingFace (`hexgrad/Kokoro-82M`). Verify sentence-level streaming works. Test with representative LLM outputs from the RAG pipeline.

3. **Prepare 50 test audio clips** — record yourself asking 50 financial queries (the same queries used in Experiments 1–3). Keep each clip under 10 seconds. Save as 16kHz mono WAV. These are the fixed inputs for Experiment 4.

4. **Validate the verifier integration** — Run DeBERTa-v3-base on the same LLM outputs, confirm ~40ms latency on CPU. Already planned for other experiments; just verify it works standalone.

**Exit criteria**: Moonshine + Kokoro + verifier all work locally with acceptable latency. 50 audio clips recorded.

### Phase 2: Modal App Setup

5. **Create Modal Volume** for model weights — pre-download Moonshine, Kokoro, and DeBERTa weights into a persistent Modal Volume. This eliminates download time during cold starts.

6. **Write the Modal App** (`experiment4_modal.py`) as a single `@app.cls` class:
   - `@modal.enter()`: Load all 4 models (Moonshine STT, Llama 3.1 8B via vLLM, DeBERTa verifier, Kokoro TTS) from the cached Volume
   - `@modal.method()`: `run_single(audio_bytes) -> dict` — runs one query through the full pipeline with `time.perf_counter()` at each boundary
   - `@modal.method()`: `run_single_no_verifier(audio_bytes) -> dict` — same pipeline, skip verifier (for comparison)

7. **Configure**: `gpu="A10"`, `min_containers=1`, `scaledown_window=1200`, `timeout=600`

8. **Test with 2–3 audio clips** — verify per-component timing works, no errors, audio output plays back correctly.

**Exit criteria**: Modal App runs end-to-end on 2–3 test clips with correct per-component latency measurements.

### Phase 3: Run Experiment 4

9. **Run 50 queries WITH verifier** — call `run_single.remote()` for each audio clip. Collect: `stt_ms`, `llm_ms`, `verifier_ms`, `tts_ms`, `total_ms`, `faithfulness_score`

10. **Run 50 queries WITHOUT verifier** — call `run_single_no_verifier.remote()` for the same 50 clips. Collect: `stt_ms`, `llm_ms`, `tts_ms`, `total_ms`

11. **Save results** to `results/experiment4_with_verifier.json` and `results/experiment4_without_verifier.json`

**Exit criteria**: 100 result records (50 with verifier, 50 without). Each record has per-component timing.

### Phase 4: Analysis & Paper Results

12. **Compute statistics**:
    - Median, P95, P99 for each component and total
    - **Overhead = median(with_verifier) − median(without_verifier)** — this is the key number
    - Verify overhead ≈ 40–50ms (verifier's contribution)
    - Compare against the <50ms target stated in the paper

13. **Generate paper table**:

| Component | Median (ms) | P95 (ms) |
|-----------|------------|----------|
| STT (Moonshine Small) | ~120 | ~200 |
| RAG + LLM (Llama 3.1 8B) | ~800 | ~1200 |
| Faithfulness Verifier (DeBERTa) | ~40 | ~50 |
| TTS (Kokoro-82M) | ~50 | ~80 |
| **Total with verifier** | **~1010** | **~1530** |
| **Total without verifier** | **~970** | **~1480** |
| **Verifier overhead** | **~40** | **~50** |

14. **Write the deployability argument**: "The verification layer adds ~40ms median overhead (50ms P95) to a ~1s end-to-end pipeline — a 4% increase that is imperceptible in real-time voice interaction and well within the latency budget for conversational AI."

### Phase 5: Reproducibility Packaging

15. **Pin all model versions** in the experiment script:
    - `moonshine-ai/moonshine-small` @ specific commit
    - `hexgrad/Kokoro-82M` @ specific commit
    - `cross-encoder/nli-deberta-v3-base` @ specific commit
    - `meta-llama/Llama-3.1-8B-Instruct` @ specific commit

16. **Provide reproduction instructions** in the paper appendix and repo README:
    - `pip install modal && modal token set`
    - `modal volume put faithfulvoice-models <weights_dir>`
    - `modal run experiment4_modal.py`
    - One command to reproduce all 100 runs

17. **Upload audio clips** to HuggingFace alongside the dataset — the 50 test audio clips become part of the released artifact.

---

## Cost Estimate

| Resource | Duration | Modal Price | Cost |
|----------|----------|-------------|------|
| A10 GPU (model warm) | ~30 min | $1.10/hr | $0.55 |
| A10 GPU (50 runs × 2) | ~30 min | $1.10/hr | $0.55 |
| CPU (verifier only baseline) | ~5 min | $0.047/core/hr | ~$0.01 |
| **Total** | | | **~$1.10** |

With $10,000 academic credits: **$0**.

---

## What This Plan Does NOT Cover

- Experiments 1–3, 5 (text-only, already planned in RESEARCH.md)
- Human evaluation
- Paper writing
- Dataset generation
- Fine-tuning the verifier
- Audio quality evaluation (not needed — the paper doesn't evaluate voice quality)

This plan is **only** for Experiment 4: proving the verifier's latency overhead is acceptable for real-time deployment.

---

## Model Reference Card

| Role | Model | Version | License | VRAM | Latency |
|------|-------|---------|---------|------|---------|
| STT | Moonshine Small | `moonshine-ai/moonshine-small` | MIT | ~150MB | 73–165ms |
| LLM | Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 Community | ~6GB (4-bit) | ~800ms |
| Verifier | DeBERTa NLI Base | `cross-encoder/nli-deberta-v3-base` | MIT | ~350MB | ~40ms |
| TTS | Kokoro 82M | `hexgrad/Kokoro-82M` | Apache-2.0 | <500MB | 30–80ms |

All models are self-hostable. No API keys required. No proprietary dependencies. Fully reproducible.

### Implementation: Pure Python

The entire pipeline — STT, LLM, verifier, TTS — runs in a single Python process on Modal. No Rust. No subprocess dispatch. No ONNX Runtime bindings. No state machine. Just:

```python
async def run_pipeline(audio: bytes, use_verifier: bool = True) -> dict:
    t0 = time.perf_counter()
    text = moonshine.transcribe(audio)           # ~120ms
    t1 = time.perf_counter()
    answer, chunks = vllm.generate(text)         # ~800ms
    t2 = time.perf_counter()
    score = deberta.verify(answer, chunks) if use_verifier else None  # ~40ms
    t3 = time.perf_counter()
    audio_out = kokoro.synthesize(answer)         # ~50ms
    t4 = time.perf_counter()
    return {"stt_ms": (t1-t0)*1000, "llm_ms": (t2-t1)*1000,
            "verifier_ms": (t3-t2)*1000, "tts_ms": (t4-t3)*1000, ...}
```

Every model has a pip-installable Python package. Every model loads from HuggingFace weights. The Modal App is ~100 lines. Another researcher reproduces with one command.
