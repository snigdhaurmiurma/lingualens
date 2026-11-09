# LinguaLens: Multilingual Search Fairness & Evaluation Platform

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js: 14+](https://img.shields.io/badge/Next.js-14+-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

> **An empirical research testbed and diagnostic platform evaluating cross-lingual semantic search parity, tokenization asymmetry, and transliteration penalties across diverse scripts.**

---

## 🌟 Executive Summary

Dense vector embedding models (bi-encoders) are now the backbone of modern semantic search, RAG pipelines, and recommendation engines. However, models that report stellar multilingual retrieval metrics frequently suffer from **severe hidden disparities** when queries deviate from high-resource linguistic norms:
- **Native Non-Latin Scripts:** Bengali (`bn`, `Beng`), Hindi (`hi`, `Deva`), Arabic (`ar`, `Arab`).
- **Phonetic Transliterations:** Banglish (`bn-Latn`), Romanized Hindi (`hi-Latn`).
- **Tokenization Explosions:** Subword fertility ratios 3x–5x higher in low-resource and transliterated vernaculars, causing vector drift and retrieval collapse.

**LinguaLens** provides an open-source, scientifically validated platform that decouples **overall model retrieval quality** from **inter-language performance disparity**, empowering researchers and engineers to detect, quantify, and mitigate search fairness gaps.

---

## 🔬 Core Capabilities

| Capability | Description |
|---|---|
| **Dual Disparity Formulation** | Measures both Absolute Performance Gap ($\Delta_{abs}$) and Relative Disparity Ratio ($D_{rel}$) using paired semantic query groups ($Q_G$). |
| **Statistical Rigor** | Computes empirical 95% Bootstrap Confidence Intervals ($B=1000$) and paired Wilcoxon signed-rank significance tests ($p < 0.05$). |
| **Transliteration Penalty ($\Phi_{\text{trans}}$)** | Quantifies retrieval collapse when users switch between native scripts and Latin-script phonetic spellings (e.g. `bn` vs `bn-Latn`). |
| **Tokenizer Diagnostic Inspector** | Audits subword fragmentation, token IDs, and fertility ratios across vocabulary sets. |
| **Dual Execution Engine** | Runs in full-scale PostgreSQL + `pgvector` + Redis mode OR zero-external-dependency embedded mode (SQLite + in-memory NumPy matrix). |
| **Hardware Agnostic** | Tuned for CPU baseline inference (e.g., 4 cores, 8GB RAM) with optional CUDA GPU acceleration. |

---

## 🏛 Architecture Overview

```
                      +-----------------------------+
                      |   Next.js 14 Web Frontend   |
                      |  (Dashboard, SERP Diff Lab) |
                      +--------------+--------------+
                                     | REST / SSE
                                     v
                      +-----------------------------+
                      |   FastAPI Gateway Service   |
                      | (Validation, Job Scheduler) |
                      +--------------+--------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
+-----------------------------+             +-----------------------------+
|    ML Inference Adapters    |             |   Dual Vector Persistence   |
| (E5-Small, MiniLM, MuRIL)   |             | (PostgreSQL+pgvector /      |
|  - Tokenizer Inspector      |             |  In-Memory NumPy Cosine)    |
|  - IR Metrics (nDCG@K, MRR) |             +-----------------------------+
+-----------------------------+
```

---

## 🚀 Quickstart (Local Development)

### Prerequisites
- Python 3.11+ (managed via [`uv`](https://docs.astral.sh/uv/) or standard `venv`)
- Node.js 20+ and npm / pnpm
- Git

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-org/LinguaLens.git
cd LinguaLens
cp .env.example .env
```

### 2. Backend Setup (using `uv`)
```bash
cd backend
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
pytest
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the Evaluation Dashboard.

---

## 📚 Technical Documentation

Comprehensive architectural and research documentation is available in the repository:
- [Software Requirements Specification (SRS)](docs/SRS.md)
- [System Architecture & Design](docs/architecture.md)
- [Database Schema & ER Model](docs/database-schema.md)
- [REST & Streaming API Specification](docs/api-specification.md)
- [Implementation Roadmap & Milestones](docs/implementation-roadmap.md)
- [Research Methodology & Mathematical Formulations](research/methodology.md)
- [Embedding Model Registry & Audit](research/model-registry.md)
- [Methodological Limitations](research/limitations.md)

---

## ⚖ Citation & Scientific Integrity

If you use LinguaLens in your research or production auditing, please cite:

```bibtex
@software{lingualens2026,
  author = {LinguaLens Contributors},
  title = {LinguaLens: Multilingual Search Fairness & Evaluation Platform},
  year = {2026},
  url = {https://github.com/your-org/LinguaLens}
}
```

## 📄 License
This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
