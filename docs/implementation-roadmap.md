# LinguaLens: Phased Implementation Roadmap

**Document Version:** 1.0.0  
**Project Cadence:** Test-Driven, Milestone-Based  
**Date:** November 2026  

---

## 1. Roadmap Overview & Timeline

The project is structured into seven distinct, incremental milestones. Each milestone delivers an independently testable, functional artifact before moving forward.

```mermaid
gantt
    title LinguaLens Development Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1: Core Engine
    Milestone 0 - Technical Planning & Architecture      :done, m0, 2026-11-09, 1d
    Milestone 1 - IR Metrics & Evaluation Engine         :active, m1, 2026-11-10, 3d
    Milestone 2 - Model Adapter Registry & Inference     :m2, after m1, 4d
    section Phase 2: Persistence & API
    Milestone 3 - Database & Dual Vector Storage         :m3, after m2, 3d
    Milestone 4 - FastAPI REST & Streaming Gateway       :m4, after m3, 4d
    section Phase 3: UI & Analytics
    Milestone 5 - Next.js UI & Multilingual Query Lab    :m5, after m4, 5d
    Milestone 6 - Fairness Analyzer & Bootstrap Engine   :m6, after m5, 4d
    Milestone 7 - Report Generator, Docker & CI/CD       :m7, after m6, 3d
```

---

## 2. Milestone Breakdown & Acceptance Criteria

### Milestone 0: Technical Planning, Architecture & Specification (Current)
- **Objective:** Finalize SRS, system architecture, database schema, API contracts, research methodology, model registry audit, and repository skeleton.
- **Deliverables:** Complete documentation under `docs/` and `research/`, project root configurations (`.gitignore`, `.env.example`, `docker-compose.yml`, `README.md`).
- **Acceptance Criteria:**
  - All specification documents pass peer review standards.
  - Candidate models verified for license compatibility and hardware bounds.
  - Zero unvetted external dependencies.

---

### Milestone 1: Pure Domain IR Evaluation Engine (COMPLETED)
- **Objective:** Implement the pure Python evaluation engine calculating $P@K, R@K, \text{MRR}, \text{nDCG}@K$, Absolute Gap ($\Delta_{abs}$), and Relative Disparity ($D_{rel}$) with graded relevance ($0 \le r \le 3$).
- **Key Modules:** `backend/app/core/metrics.py`, `backend/app/core/fairness.py`, `backend/app/core/bootstrap.py`.
- **Dependencies:** `numpy`, `scipy`.
- **Status:** **Verified & Complete.** 17/17 automated Pytest unit tests passing with zero failures or warnings. Covers analytical precision, rank discounting, non-parametric percentile bootstrapping, and paired Wilcoxon hypothesis testing.

---

### Milestone 2: Model Adapter Registry & Embedding Inference
- **Objective:** Build the pluggable `BaseEmbeddingAdapter` interface and concrete implementations for `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-small`, and `muril-base-cased`.
- **Key Modules:** `backend/app/adapters/`, `backend/app/services/inference_service.py`, `backend/app/services/tokenizer_service.py`.
- **Acceptance Criteria:**
  - Seamless CPU inference execution with batching.
  - Strict enforcement of model-specific prefixes (e.g., `"query: "` / `"passage: "` for E5).
  - Subword fertility ratio and token inspection calculation for Bengali, Banglish, Hindi, and Arabic.

---

### Milestone 3: Storage Layer & Dual-Mode Vector Retrieval
- **Objective:** Implement SQLAlchemy 2.0 declarative database entities, Alembic migrations, and the dual-mode vector retrieval engine (`PgVectorRetriever` + `NumPyRetriever`).
- **Key Modules:** `backend/app/models/`, `backend/app/services/retrieval_service.py`, `alembic/`.
- **Acceptance Criteria:**
  - Database migrations generate clean schema on PostgreSQL and SQLite.
  - In-memory `NumPyRetriever` achieves $<15\text{ms}$ query latency across 10,000 document vectors on CPU.

---

### Milestone 4: FastAPI REST & Streaming Gateway
- **Objective:** Expose validated HTTP endpoints and Server-Sent Events (SSE) for experiment tracking, Query Lab search, and dataset management.
- **Key Modules:** `backend/app/api/v1/`, `backend/app/schemas/`.
- **Acceptance Criteria:**
  - OpenAPI 3.1 documentation generated automatically.
  - Streaming progress updates via SSE without socket leakage.
  - Comprehensive integration test suite using `httpx` and `pytest-asyncio`.

---

### Milestone 5: Next.js Frontend Dashboard & Multilingual Query Lab
- **Objective:** Build a research-grade UI with Next.js 14, Tailwind CSS, shadcn/ui, and Recharts.
- **Key Features:**
  - Evaluation Dashboard with interactive radar and metric bar charts.
  - Side-by-side Multilingual SERP comparison with rank shift badges.
  - Tokenizer inspector showing subword splits across script variants.
  - Dark/Light mode theme system using Midnight Navy (`#0A0F1D`) and Electric Violet accents.
- **Acceptance Criteria:**
  - Full keyboard accessibility and WCAG AA contrast compliance.
  - Correct font rendering for Indic scripts (Bengali, Devanagari) and RTL Arabic.
  - Responsive layout for desktop and tablet screens.

---

### Milestone 6: Language Fairness Analyzer & Bootstrap Engine
- **Objective:** Complete the statistical fairness suite with non-parametric bootstrap resampling, paired Wilcoxon signed-rank tests, and disparity heatmaps.
- **Acceptance Criteria:**
  - Interactive confidence interval error bar visualizations.
  - Explicit Transliteration Penalty reporting ($\Phi_{\text{trans}}$).
  - Explicit warning badges when running on `DEMONSTRATION` data.

---

### Milestone 7: Research Report Generator, Datasets & Production Packaging
- **Objective:** Implement PDF/Markdown research report compilation, package demonstration and validated benchmark datasets, configure Docker Compose, and finalize GitHub Actions CI.
- **Acceptance Criteria:**
  - Single-command startup via `docker compose up --build`.
  - Zero-dependency local developer run mode (`uv run backend` + `npm run dev`).
  - Automated CI matrix testing Python 3.11/3.12 and Node 20+.
