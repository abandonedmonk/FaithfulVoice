# FaithfulVoice

**Real-Time Faithfulness Verification for Voice-Driven Financial RAG Systems**

A state-of-the-art system that combines voice interaction with retrieval-augmented generation (RAG) to provide accurate, trustworthy answers about financial documents. FaithfulVoice ensures that responses are grounded in factual SEC filings and detects hallucinations in real-time.

---

## 📋 Overview

FaithfulVoice is a comprehensive pipeline designed to:
- **Ingest** SEC EDGAR financial filings with intelligent document cleaning and section-aware chunking
- **Retrieve** relevant financial data using vector embeddings and semantic search
- **Verify** response faithfulness against source documents before providing answers
- **Deliver** verified information through natural voice interaction (STT → RAG → Verifier → TTS)

This system is designed for financial analysts, investors, and applications requiring high-confidence information extraction from financial documents.

---

## ✨ Key Features

- **Intelligent Document Processing**: Automated cleaning of EDGAR HTML filings with section extraction and hierarchical chunking
- **Voice-Driven Interface**: End-to-end voice pipeline with Moonshine STT and Kokoro TTS
- **Faithfulness Verification**: Real-time verification of retrieved information against source documents
- **Vector Search**: Fast semantic search using FastEmbed and Qdrant vector database
- **Multi-Document Querying**: Support for 10+ financial ticker symbols with query dataset generation
- **Comprehensive Experiment Suite**: Evaluates accuracy, latency, hallucination rates, and retrieval quality
- **Modular Architecture**: LangGraph-based node system for flexible pipeline composition

---

## 🔧 Prerequisites

Before getting started, ensure you have:
- **Python 3.9+** installed
- **Docker & Docker Compose** for running Qdrant
- **Git** for version control
- **4GB+ RAM** recommended for embeddings and LLM inference
- **Internet connection** for downloading SEC filings from EDGAR

---

## 📦 Installation

### 1. Clone and Navigate

```bash
cd FaithfulVoice
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Settings

Edit `config.yaml` to customize:
- Model selections (embeddings, LLM, verifier)
- Qdrant connection details
- Chunking strategy parameters
- Retrieval settings

```yaml
# Example config.yaml
embeddings:
  model: "BAAI/bge-small-en-v1.5"

qdrant:
  host: "localhost"
  port: 6333

chunking:
  chunk_size: 512
  overlap: 50
```

---

## 📂 Project Structure

```
FaithfulVoice/
├── src/                          # Core RAG + Verifier pipeline
│   ├── cleaner.py               # EDGAR HTML cleaning, TOC map, section extraction
│   ├── parser.py                # unstructured wrapper, section routing
│   ├── chunker.py               # section-aware parent/child chunking + table markdown
│   ├── metadata.py              # metadata extraction from filing filenames
│   ├── ingest.py                # Qdrant batch upsert with FastEmbed
│   ├── nodes/                   # LangGraph nodes (planned)
│   └── voice/                   # Production voice pipeline
│       ├── stt.py               # Moonshine STT
│       ├── tts.py               # Kokoro TTS
│       └── pipeline.py          # STT → RAG → Verifier → TTS
│
├── scripts/                     # Pipeline & utility scripts
│   ├── download_filings.py      # Download SEC EDGAR filings via API
│   ├── process_all_filings.py   # Batch: clean → chunk → write JSONL
│   ├── extract_q4_from_10k.py   # Extract Q4 10-Q sections from 10-K filings
│   ├── summarize_tables.py      # LLM table summarization for embeddings
│   └── merge_queries.py         # Merge per-ticker query JSONL into one dataset
│
├── data/
│   ├── raw/                     # Downloaded EDGAR filings (.htm)
│   ├── cleaned/                 # Cleaned filings (ixbrl tags removed)
│   ├── processed/               # Chunked output (_chunks.jsonl)
│   ├── queries/                 # Per-ticker query datasets (6 types × 10 tickers)
│   ├── queries_full_dataset/    # Merged query dataset
│   └── audio/test_clips/        # WAV clips for latency experiments
│
├── results/                     # Experiment results
│   ├── exp1/ ... exp5/
│   └── human_eval/
│
├── experiments/                 # Experiment scripts (planned)
├── finetune/                    # FinFaithVerifier fine-tuning (planned)
├── dataset/                     # HuggingFace release prep (planned)
│
├── docs/
│   ├── RESEARCH.md              # Research plan & experiments
│   ├── BUILD_GUIDE.md           # Setup & build guide
│   ├── EXPERIMENTS.md           # Detailed experiment specifications
│   ├── Chunking_Strategy.md     # SEC filing chunking strategy
│   ├── RAG_Roadmap.md           # RAG pipeline build roadmap
│   ├── Dataset Generation.md    # Dataset generation methodology
│   ├── Voice_Latency_Plan.md    # Experiment 4 latency plan
│   ├── NOTES.md                 # Planning notes
│   └── outdated/                # Older doc versions
│
---

## 🚀 Quick Start

### Option 1: Full Pipeline Setup

```bash
# 1. Start Qdrant vector database
docker-compose up -d

# 2. Download SEC EDGAR filings
python scripts/download_filings.py

# 3. Process filings (clean, parse, chunk)
python scripts/process_all_filings.py

# 4. Ingest chunks into Qdrant
python -m src.ingest

# 5. Generate query datasets
python scripts/merge_queries.py
```

### Option 2: Use Pre-Processed Data

```bash
# If you already have processed data, just start services:
docker-compose up -d
python -m src.ingest
```

---

## 🛠️ Usage Guide

### Processing SEC Filings

```bash
# Download specific filings from EDGAR
python scripts/download_filings.py --tickers AAPL MSFT --years 2023 2024

# Clean and chunk all downloaded filings
python scripts/process_all_filings.py --output-dir data/processed/

# Extract quarterly sections from 10-K filings
python scripts/extract_q4_from_10k.py --input-file data/raw/10k_filing.htm

# Summarize tables for better embeddings
python scripts/summarize_tables.py --model gpt-4 --batch-size 10
```

### Voice Pipeline

```python
from src.voice.pipeline import FinancialVoiceRAG

# Initialize the complete voice pipeline
rag = FinancialVoiceRAG(config_path='config.yaml')

# Process audio query and get verified response
response, confidence, source = rag.process_audio('audio_query.wav')
print(f"Answer: {response}")
print(f"Confidence: {confidence}%")
print(f"Source: {source}")
```

### Programmatic Querying

```python
from src.ingest import QdrantClient
from src.parser import retrieve_documents

# Initialize retrieval
client = QdrantClient()

# Search for relevant documents
query = "What were Apple's Q4 2023 earnings?"
results = client.search(query, top_k=5)

# Results include document chunks, confidence scores, and metadata
for result in results:
    print(f"Score: {result['score']}")
    print(f"Content: {result['content']}")
    print(f"Source: {result['metadata']['source']}")
```

---

## 📚 Core Components

### Source Code (`src/`)

| File | Purpose |
|------|---------|
| `cleaner.py` | EDGAR HTML cleaning, TOC mapping, and section extraction |
| `parser.py` | Document parsing with unstructured library and section routing |
| `chunker.py` | Hierarchical parent/child chunking with table markdown conversion |
| `metadata.py` | Metadata extraction from SEC filing filenames and content |
| `ingest.py` | Batch upsert to Qdrant with FastEmbed embeddings |
| `voice/stt.py` | Speech-to-text using Moonshine model |
| `voice/tts.py` | Text-to-speech using Kokoro model |
| `voice/pipeline.py` | End-to-end voice RAG pipeline with verification |

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `download_filings.py` | Download SEC EDGAR filings via REST API |
| `process_all_filings.py` | Batch processing: clean → parse → chunk → JSONL |
| `extract_q4_from_10k.py` | Extract quarterly sections from annual filings |
| `summarize_tables.py` | LLM-based table summarization for embeddings |
| `merge_queries.py` | Combine per-ticker query datasets into unified dataset |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Voice Input (Audio)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────▼─────┐
                    │   STT     │ Moonshine
                    │ (Audio→   │
                    │  Text)    │
                    └────┬─────┘
                         │
                    ┌────▼────────────┐
                    │  Query          │
                    │  Embedding      │
                    └────┬────────────┘
                         │
     ┌───────────────────┼──────────────────┐
     │                   │                  │
┌────▼────────┐  ┌──────▼──────┐   ┌──────▼──────┐
│  Qdrant      │  │  Retrieval  │   │  Ranking    │
│  Vector DB   │  │  Engine     │   │  Layer      │
└────┬────────┘  └──────┬──────┘   └──────┬──────┘
     │                  │                  │
     └──────────────────┼──────────────────┘
                        │
                   ┌────▼──────────┐
                   │  LLM RAG      │
                   │  Generation   │
                   └────┬──────────┘
                        │
                   ┌────▼──────────────┐
                   │  Faithfulness     │
                   │  Verification     │
                   └────┬──────────────┘
                        │
                   ┌────▼──────────┐
                   │   TTS         │ Kokoro
                   │  (Text→       │
                   │   Audio)      │
                   └────┬──────────┘
                        │
                 ┌──────▼────────┐
                 │  Voice Output  │
                 └────────────────┘
```

---

## 📊 Data Organization

```
data/
├── raw/                         # Downloaded EDGAR filings (.htm)
├── cleaned/                     # Processed filings (tags removed)
├── processed/                   # Chunked JSONL output
├── queries/                     # Per-ticker datasets (6 types × 10 companies)
├── queries_full_dataset/        # Merged unified query dataset
└── audio/test_clips/            # WAV files for latency testing
```

---

## 🧪 Experiments

FaithfulVoice includes comprehensive evaluation suites:

- **Exp 1-2**: Faithfulness verification accuracy and metrics
- **Exp 3**: Retrieval quality and ranking evaluation
- **Exp 4**: Voice latency measurements and optimization
- **Exp 5**: Multi-hop reasoning and complex query handling
- **Human Eval**: Manual evaluation of response quality

Results are stored in `results/` directory with detailed metrics and visualizations.

---

## 📖 Documentation

- [Research Plan](docs/RESEARCH.md) — Project vision, hypotheses, and research questions
- [Build Guide](docs/BUILD_GUIDE.md) — Detailed setup and environment configuration
- [Experiment Specs](docs/EXPERIMENTS.md) — Formal experiment definitions and evaluation metrics
- [Chunking Strategy](docs/Chunking_Strategy.md) — SEC filing chunking methodology
- [RAG Roadmap](docs/RAG_Roadmap.md) — RAG pipeline development roadmap
- [Dataset Generation](docs/Dataset%20Generation.md) — Query dataset creation methodology
- [Voice Latency Plan](docs/Voice_Latency_Plan.md) — Latency analysis and optimization strategies

---

## ⚙️ Configuration

### `config.yaml` Reference

```yaml
embeddings:
  provider: "huggingface"
  model: "BAAI/bge-small-en-v1.5"
  batch_size: 32

qdrant:
  host: "localhost"
  port: 6333
  collection_name: "financial_filings"
  vector_size: 384

chunking:
  chunk_size: 512
  chunk_overlap: 50
  min_chunk_size: 100

retrieval:
  top_k: 5
  similarity_threshold: 0.7

models:
  llm: "gpt-4"
  verifier: "FinFaithVerifier"
  
voice:
  stt_model: "moonshine"
  tts_model: "kokoro"
```

---

## 🔍 Troubleshooting

### Qdrant Connection Issues

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart

# View logs
docker-compose logs qdrant
```

### Memory and Performance

- **Out of Memory**: Reduce `batch_size` in config.yaml
- **Slow Embeddings**: Use smaller model (e.g., `all-MiniLM-L6-v2`)
- **High Latency**: Process filings in smaller batches

### Missing Dependencies

```bash
# Reinstall all requirements
pip install --upgrade -r requirements.txt

# Install specific packages
pip install unstructured langchain qdrant-client pyaudio
```

---

## 📝 Query Dataset Types

The system supports query generation across 6 question types:

1. **Direct Fact** - Simple factual questions about document content
2. **False Premise** - Questions with incorrect assumptions to test verification
3. **Multi-hop** - Questions requiring multiple document traversals
4. **Out-of-Scope** - Questions outside document scope (hallucination detection)
5. **Qualitative** - Opinion and trend-based questions
6. **Temporal** - Time-dependent questions about financial periods

Datasets cover major financial companies: AAPL, MSFT, GOOGL, AMZN, META, TSLA, JPM, and others.

---

## 🚀 Performance Optimization

### For Large-Scale Processing

1. **Batch Processing**: Use `process_all_filings.py` for efficient bulk processing
2. **Parallel Ingestion**: Qdrant supports concurrent upserts
3. **Model Selection**: Balance accuracy vs. speed with model choice
4. **Caching**: Enable query result caching for repeated questions

### Voice Pipeline Optimization

1. **STT Streaming**: Process audio chunks for lower latency
2. **LLM Batching**: Group multiple queries for batch inference
3. **TTS Caching**: Cache synthesized audio for common phrases

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and test thoroughly
3. Follow PEP 8 code style
4. Add docstrings to functions and modules
5. Include unit tests for new features
6. Submit a pull request with clear description

---

## 📋 Quick Reference

| Task | Command |
|------|---------|
| Start Qdrant | `docker-compose up -d` |
| Stop Qdrant | `docker-compose down` |
| Download filings | `python scripts/download_filings.py` |
| Process filings | `python scripts/process_all_filings.py` |
| Ingest data | `python -m src.ingest` |
| Run experiments | `python experiments/run_all.py` |
| View logs | `docker-compose logs -f` |
| Test voice | `python src/voice/pipeline.py --test` |

---

## 📞 Support & Resources

For questions and issues:
- Review [Documentation](#-documentation) section
- Check [Troubleshooting](#-troubleshooting) section
- Open an issue with error logs and reproduction steps
- Refer to code comments for implementation details

---

## 📜 License

This project is part of the FinNLP research initiative.

---

## 🔗 Related Projects

- **FinNLP Dataset**: Financial NLP benchmark datasets
- **FinFaithVerifier**: Faithfulness verification model
- **FaithfulRAG**: RAG system for trustworthy information extraction

---

**Status**: Active Development  
**Last Updated**: June 7, 2026  
**Maintainers**: FinNLP Research Team
