# LinguaLens: Software Requirements Specification (SRS)

**Document Version:** 1.0.0  
**Status:** Approved for Foundation Architecture  
**Author:** LinguaLens Core Engineering & Research Team  
**Date:** November 2026  

---

## 1. Introduction

### 1.1 Purpose
This document provides the formal Software Requirements Specification (SRS) for **LinguaLens**, an open-source, research-backed Multilingual Search Fairness & Evaluation Platform. It defines the functional, non-functional, interface, and behavioral requirements necessary to evaluate, audit, and benchmark multilingual semantic search and text embedding models across diverse linguistic typologies and script systems.

### 1.2 Scope and Product Vision
Modern information retrieval systems increasingly rely on dense semantic vector embeddings. While contemporary multilingual models claim broad cross-lingual parity across dozens of languages, their real-world performance exhibits stark degradation when queries deviate from high-resource linguistic norms, specifically:
- Non-Latin native scripts (e.g., Bengali `bn`, Hindi `hi`, Arabic `ar`).
- Phonetic transliteration into Latin scripts (e.g., Banglish `bn-Latn`, Romanized Hindi `hi-Latn`).
- Directionality transitions (Right-to-Left Arabic vs. Left-to-Right Indo-Aryan and Germanic scripts).
- Morphological complexity and agglutinative compounds.

LinguaLens provides an empirical, reproducible testbed that systematically decouples **overall model retrieval quality** from **inter-language performance disparity**, providing machine learning engineers, search architects, and researchers with actionable diagnostic metrics.

### 1.3 Definitions, Acronyms, and Abbreviations
- **IR:** Information Retrieval.
- **Bi-Encoder:** Architecture encoding queries and documents into isolated vector representations compared via vector similarity (e.g., dot-product, cosine).
- **Cross-Encoder:** Architecture performing joint attention over query and document tokens simultaneously.
- **Query Group ($Q_G$):** A semantically equivalent cluster of queries expressing the exact same information need in different languages or script variants.
- **Graded Relevance ($r \in \{0, 1, 2, 3\}$):** Multi-level human or authoritative relevance judgments (0 = Irrelevant, 1 = Marginally Relevant, 2 = Relevant, 3 = Highly Relevant).
- **Banglish (`bn-Latn`):** Informal or phonetically spelled Bengali language typed using ASCII/Latin script characters without standardized orthography.
- **Disparity Metric:** Statistical quantification of retrieval effectiveness divergence between a privileged language baseline (typically English) and target vernacular variants under identical semantic intent.

---

## 2. Overall Description

### 2.1 Product Perspective
LinguaLens operates as a standalone research and diagnostic platform. It consists of a decoupled architecture with:
1. A Next.js (TypeScript) analytical research dashboard.
2. A high-performance Python (FastAPI) core computational backend.
3. A pluggable embedding model runner capable of local CPU execution with optional GPU/CUDA acceleration.
4. A relational persistence layer (PostgreSQL with optional `pgvector` extension) with a zero-external-dependency fallback (NumPy cosine similarity / DuckDB / SQLite).
5. Asynchronous task queue architecture (Redis + Celery or FastAPI BackgroundTasks).

```
+--------------------------------------------------------------+
|                    Next.js Web Interface                     |
|  (Evaluation Dashboard, Query Lab, Model Benchmarker, etc.)  |
+------------------------------+-------------------------------+
                               | REST / SSE / WebSocket
                               v
+--------------------------------------------------------------+
|                   FastAPI Backend Gateway                    |
|  (Auth, Request Validation, Route Dispatch, Task Scheduling) |
+------------------------------+-------------------------------+
           |                                       |
           v                                       v
+-----------------------+               +----------------------+
| ML Inference & Engine |               | Data & Vector Store  |
| - Model Adapters      |               | - PostgreSQL/SQLite  |
| - Tokenizer Auditing  |               | - pgvector / NumPy   |
| - Metric Computations |               | - Alembic Migration  |
+-----------------------+               +----------------------+
```

### 2.2 User Personas
- **ML / Search Engineer:** Evaluates candidate embedding models (e.g., BGE-M3, Multilingual-E5, IndicBERT, paraphrase-multilingual-MiniLM) to assess production search fairness before deploying semantic search into multilingual geographic regions.
- **NLP / IR Researcher:** Formulates controlled experiment runs, tests hypotheses regarding tokenization fragmentation and script bias, computes bootstrap confidence intervals, and exports LaTeX/PDF research reports.
- **Product Trust & Fairness Auditor:** Audits existing enterprise retrieval models to identify disparate access to critical information across non-English speaking user cohorts.

---

## 3. Specific Functional Requirements

### 3.1 Module 1: Evaluation Dashboard
- **FR-1.1:** Display high-level system overview: indexed documents, benchmark datasets, configured models, active experiments, and aggregate disparity scores.
- **FR-1.2:** Render radar charts, parallel coordinate plots, and tabular breakdowns comparing metrics ($nDCG@K$, $MRR$, $P@K$, $R@K$, Latency) across language groups.
- **FR-1.3:** Allow interactive filtering by dataset version, model identifier, cutoff threshold $K \in [1, 5, 10, 20, 50]$, and metric type.

### 3.2 Module 2: Multilingual Query Lab
- **FR-2.1:** Allow interactive execution of a single semantic intent expressed across multiple languages (e.g., query typed in English, Bengali, Banglish, Hindi, and Arabic).
- **FR-2.2:** Execute concurrent dense vector retrieval against the indexed document collection for all language variants.
- **FR-2.3:** Display a side-by-side comparative SERP (Search Engine Results Page) highlighting rank shifts, document overlap, and retrieved relevance scores.
- **FR-2.4:** Provide a Tokenizer Inspector revealing subword token splits, token counts, and unknown/OOV token frequency across script variants.

### 3.3 Module 3: Embedding Model Benchmarking
- **FR-3.1:** Maintain a pluggable Model Registry supporting Hugging Face Transformers and Sentence Transformers.
- **FR-3.2:** Execute batch inference pipelines with configurable batch sizes, sequence lengths, normalization options, and device targets (`cpu`, `cuda`).
- **FR-3.3:** Record micro-benchmarks: latency per query (mean, p50, p95, p99), memory consumption (peak RAM / VRAM), and throughput (queries/sec).

### 3.4 Module 4: Language Fairness Analyzer
- **FR-4.1:** Compute Absolute Performance Gap ($\Delta_{abs} = M_{\text{baseline}} - M_{\text{target}}$) and Relative Disparity Ratio ($D_{rel} = \frac{M_{\text{target}}}{M_{\text{baseline}}}$) across paired query groups.
- **FR-4.2:** Perform bootstrapping ($N=1000$ iterations) to calculate 95% Confidence Intervals (CI) for all disparity metrics.
- **FR-4.3:** Provide statistical hypothesis testing (paired Wilcoxon signed-rank test and permutation tests) to verify whether observed differences are statistically significant ($p < 0.05$).
- **FR-4.4:** Specifically assess Transliteration Penalty: performance degradation between native script (e.g., `bn`) and Latin transliteration (`bn-Latn`).

### 3.5 Module 5: Dataset Studio
- **FR-5.1:** Manage versioned benchmark datasets conforming to the standardized LinguaLens schema.
- **FR-5.2:** Enforce validation rules on upload (JSONL, CSV, Parquet): prevent dangling query IDs, verify language/script ISO codes, validate relevance scale ($0 \le r \le 3$).
- **FR-5.3:** Explicitly mark datasets as either `DEMONSTRATION` (synthetic/sample) or `VALIDATED_BENCHMARK` (human annotated / peer-reviewed).
- **FR-5.4:** Provide query group curation tools to link semantically equivalent queries across languages.

### 3.6 Module 6: Experiment Management
- **FR-6.1:** Define immutable Experiment configurations binding: Dataset Version + Model Revision + Retrieval Parameters ($K$, similarity metric, query prefix) + Hardware Environment.
- **FR-6.2:** Track execution status (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`).
- **FR-6.3:** Persist full run artifacts, environment hashes, git commit hash, random seed, and complete raw evaluation results.

### 3.7 Module 7: Research Report Generator
- **FR-7.1:** Generate exportable research reports in Markdown, JSON, and printable HTML/PDF.
- **FR-7.2:** Embed data provenance tables, statistical test summaries, methodology citations, and clear limitation disclaimers.

---

## 4. Non-Functional Requirements (NFR)

### 4.1 Performance & Resource Efficiency
- **NFR-1.1:** The backend must boot and run inference on standard commodity x86_64 / ARM64 CPUs with a minimum of 4 CPU cores and 8GB RAM.
- **NFR-1.2:** Single query vector encoding on CPU for models under 150M parameters (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) must complete in $\le 85\text{ms}$.
- **NFR-1.3:** Vector similarity search over 50,000 documents must return results in $\le 15\text{ms}$ using in-memory indexed arrays or pgvector IVFFlat/HNSW indexes.

### 4.2 Reliability, Reproducibility & Scientific Integrity
- **NFR-2.1:** Evaluation runs under identical seed, dataset version, and model revision must produce deterministic metric scores ($\pm 10^{-6}$ precision difference).
- **NFR-2.2:** Demonstration data must never be mixed with empirical benchmark metrics. The UI and API must return an explicit warning flag on demo data runs.
- **NFR-2.3:** System errors during batch evaluation must capture traceback, model state, and progress checkpoint without corrupting previous run data.

### 4.3 Security & Safety
- **NFR-3.1:** All file uploads (datasets, model weights) must be validated for MIME type, file size limits ($\le 100\text{MB}$ default), and sanitized paths to prevent directory traversal.
- **NFR-3.2:** No unauthenticated command execution or dynamic code evaluation (`eval`, `exec`) permitted.
- **NFR-3.3:** Zero transmission of proprietary queries or documents to third-party commercial APIs. All computation executes locally.

### 4.4 Usability & Accessibility
- **NFR-4.1:** Adherence to WCAG 2.1 AA contrast ratios for all UI elements in both Light and Dark themes.
- **NFR-4.2:** Native script rendering verification: correct font ligature display for Bengali (`অ-হ`), Devanagari (`अ-ह`), and right-to-left bidirectional flow for Arabic (`ابتث`).

---

## 5. External Interface Requirements

### 5.1 User Interfaces
- Modern responsive web interface implemented using Next.js 14+ App Router, Tailwind CSS, and shadcn/ui.
- Inter typography with full Unicode support (including Noto Sans Bengali, Noto Sans Devanagari, Noto Sans Arabic fallback fonts).

### 5.2 Software Interfaces
- **Database:** PostgreSQL 16+ with `pgvector` extension; SQLite fallback for portable single-machine execution.
- **Backend Framework:** FastAPI with ASGI server (Uvicorn).
- **Model Runtimes:** PyTorch 2.x, Hugging Face Transformers, Sentence-Transformers.
- **Cache/Queue:** Redis 7+ and Celery (optional in single-process developer mode).
