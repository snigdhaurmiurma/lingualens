# LinguaLens: REST & Streaming API Specification

**Document Version:** 1.1.0  
**Base URL:** `/api/v1`  
**Protocol:** HTTP/1.1 & HTTP/2, Server-Sent Events (SSE) for streaming  
**Content-Type:** `application/json` (UTF-8)

---

## 1. Architectural Overview & Design Principles

The LinguaLens REST Gateway is a FastAPI application exposing the domain core, model registry, vector retrieval engine, and statistical evaluation services to the frontend dashboard and CLI consumers.

Key principles:
1. **Thin Gateway Layer:** Routes delegate domain logic to underlying services (`DatasetService`, `RetrievalService`, `ExperimentService`, `TokenizerInspector`).
2. **Resource Attribution & Schema Validation:** Strictly enforced using Pydantic v2 schemas with `strict=False` input coercion, RFC-compliant ISO 639-1 language codes, and ISO 15924 script codes.
3. **Graceful Degradation & Safe Concurrency:**
   - Long-running evaluation tasks run non-blocking in background executors with thread-safe dedicated sessions.
   - Real-time progress updates are delivered asynchronously via Server-Sent Events (`text/event-stream`) backed by in-memory `asyncio.Queue` channels.
4. **Standardized Error Envelope:** All 4xx and 5xx responses conform to a unified error schema.

---

## 2. Error Response Standard

All errors adhere to the following JSON structure:

```json
{
  "error": true,
  "status_code": 404,
  "message": "Dataset with ID '3fa85f64-5717-4562-b3fc-2c963f66afa6' does not exist.",
  "details": null
}
```

Validation errors (HTTP 422) return detailed field breakdown:

```json
{
  "error": true,
  "status_code": 422,
  "message": "Request validation failed",
  "details": [
    {
      "field": "body.retrieval_mode",
      "issue": "Input should be 'EXACT' or 'VECTOR_INDEX'"
    }
  ]
}
```

---

## 3. Endpoints Inventory

### 3.1 Health & System Catalog

#### `GET /api/v1/health`
Return service status and database connectivity.
- **Status:** `200 OK`
- **Response:**
  ```json
  {
    "status": "healthy",
    "version": "0.1.0",
    "timestamp": "2026-11-10T02:00:00Z",
    "database": "connected"
  }
  ```

#### `GET /api/v1/system/info`
Return runtime host capability, CPU threading, and registered model list.
- **Status:** `200 OK`
- **Response:**
  ```json
  {
    "python_version": "3.11.15",
    "torch_version": "2.10.0+cpu",
    "cpu_count": 8,
    "torch_cpu_threads": 8,
    "device_default": "cpu",
    "memory_available_mb": null,
    "registered_models": [
      "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "intfloat/multilingual-e5-small"
    ]
  }
  ```

#### `GET /api/v1/models`
List available multilingual embedding adapters and metadata.
- **Status:** `200 OK`
- **Response:**
  ```json
  [
    {
      "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "display_name": "Paraphrase Multilingual MiniLM L12 v2",
      "provider": "Hugging Face / Sentence-Transformers",
      "parameter_count": 117000000,
      "embedding_dimension": 384,
      "max_sequence_length": 128,
      "license": "Apache-2.0",
      "query_prefix": "",
      "document_prefix": "",
      "supported_languages": ["en", "bn", "bn-Latn", "hi", "ar"],
      "requires_gpu": false,
      "is_loaded": false
    },
    {
      "model_id": "intfloat/multilingual-e5-small",
      "display_name": "Multilingual E5 Small",
      "provider": "Microsoft / intfloat",
      "parameter_count": 118000000,
      "embedding_dimension": 384,
      "max_sequence_length": 512,
      "license": "MIT",
      "query_prefix": "query: ",
      "document_prefix": "passage: ",
      "supported_languages": ["en", "bn", "bn-Latn", "hi", "ar"],
      "requires_gpu": false,
      "is_loaded": false
    }
  ]
  ```

---

### 3.2 Benchmark Datasets (`/api/v1/datasets`)

#### `POST /api/v1/datasets`
Create a new benchmark dataset entity.
- **Request:**
  ```json
  {
    "name": "Clinical Cross-Lingual IR Benchmark",
    "slug": "clinical-crosslingual-v1",
    "description": "Multilingual healthcare queries and passages",
    "version": "1.0.0",
    "dataset_type": "BENCHMARK",
    "license": "CC-BY-4.0",
    "source_uri": "https://example.com/data",
    "supported_languages": ["en", "bn", "bn-Latn", "hi", "ar"]
  }
  ```
- **Status:** `201 Created`

#### `GET /api/v1/datasets`
List datasets with pagination.
- **Query Params:** `skip` (int, default 0), `limit` (int, default 50).
- **Status:** `200 OK`

#### `GET /api/v1/datasets/{dataset_id}`
Retrieve dataset details by UUID.
- **Status:** `200 OK` (or `404 Not Found`)

#### `POST /api/v1/datasets/{dataset_id}/documents`
Add a document passage to the dataset.
- **Request:**
  ```json
  {
    "doc_external_id": "doc-001",
    "content": "Common symptoms of seasonal flu include sudden fever and fatigue.",
    "language_code": "en",
    "script_code": "Latn"
  }
  ```
- **Status:** `201 Created`

#### `POST /api/v1/datasets/{dataset_id}/query-groups`
Add a semantic query intent group (links equivalent queries across languages).
- **Request:**
  ```json
  {
    "canonical_intent": "Seasonal influenza symptoms",
    "domain": "healthcare"
  }
  ```
- **Status:** `201 Created`

#### `POST /api/v1/datasets/{dataset_id}/queries`
Add a language-specific query to a query group.
- **Request:**
  ```json
  {
    "query_group_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "query_text": "মৌসুমি ফ্লুর লক্ষণ",
    "language_code": "bn",
    "script_code": "Beng"
  }
  ```
- **Status:** `201 Created`

---

### 3.3 Tokenizer Inspector (`/api/v1/tokenization`)

#### `POST /api/v1/tokenization/inspect`
Inspect subword token fragmentation, sequence IDs, and subword fertility.
- **Request:**
  ```json
  {
    "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "text": "মৌসুমি ফ্লুর লক্ষণ",
    "language_code": "bn",
    "script_code": "Beng"
  }
  ```
- **Status:** `200 OK`
- **Response:**
  ```json
  {
    "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "language_code": "bn",
    "script_code": "Beng",
    "text": "মৌসুমি ফ্লুর লক্ষণ",
    "tokens": ["<s>", " মৌ", "সুমি", " ফ্লু", "র", " লক্ষণ", "</s>"],
    "token_ids": [0, 89321, 41203, 78102, 312, 65421, 2],
    "token_count": 5,
    "special_token_count": 2,
    "subword_fertility": 1.67,
    "has_unknown_tokens": false,
    "methodological_limitations": "Subword fertility measures lexical vocabulary representation efficiency; it does not constitute a comprehensive measure of semantic understanding or model quality."
  }
  ```

---

### 3.4 Semantic Vector Search (`/api/v1/search`)

#### `POST /api/v1/search`
Execute vector retrieval for an ad-hoc query against indexed dataset documents.
- **Request:**
  ```json
  {
    "dataset_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "query_text": "treatment for seasonal flu",
    "model_id": "minilm",
    "retrieval_mode": "EXACT",
    "top_k": 5,
    "language_filter": "en"
  }
  ```
- **Status:** `200 OK`
- **Response:**
  ```json
  {
    "query_text": "treatment for seasonal flu",
    "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "retrieval_mode": "EXACT",
    "latency_ms": 14.2,
    "results_count": 1,
    "results": [
      {
        "document_id": "b1b72a44-4821-4d39-9fbf-862adcf32bf1",
        "doc_external_id": "doc-001",
        "content": "Common symptoms of seasonal flu include sudden fever and fatigue.",
        "language_code": "en",
        "script_code": "Latn",
        "score": 0.8412,
        "rank": 1
      }
    ]
  }
  ```

---

### 3.5 Experiments & Evaluation Streaming (`/api/v1/experiments`)

#### `POST /api/v1/experiments`
Create an evaluation study configuration.
- **Request:**
  ```json
  {
    "name": "Cross-Lingual Disparity Study: Healthcare",
    "description": "Evaluation of MiniLM vs Multilingual-E5 on South Asian queries",
    "dataset_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "baseline_language_code": "en",
    "cutoff_k_values": [1, 5, 10, 20],
    "similarity_metric": "cosine"
  }
  ```
- **Status:** `201 Created`

#### `GET /api/v1/experiments`
List all experiments with their runs and statuses.
- **Status:** `200 OK`

#### `GET /api/v1/experiments/{experiment_id}`
Retrieve a specific experiment.
- **Status:** `200 OK`

#### `POST /api/v1/experiments/{experiment_id}/run`
Trigger an asynchronous evaluation run for a specific model adapter and retrieval mode.
- **Request:**
  ```json
  {
    "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "retrieval_mode": "EXACT",
    "random_seed": 42
  }
  ```
- **Status:** `202 Accepted`
- **Response:**
  ```json
  {
    "id": "7fa85f64-5717-4562-b3fc-2c963f66afa7",
    "experiment_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "retrieval_mode": "EXACT",
    "status": "PENDING",
    "progress_percent": 0.0,
    "error_message": null,
    "device_used": "cpu",
    "hardware_metadata": null,
    "total_latency_seconds": null,
    "created_at": "2026-11-10T02:00:00Z"
  }
  ```

#### `GET /api/v1/experiments/{experiment_id}/stream`
Stream real-time execution progress events using Server-Sent Events (SSE).
- **Media Type:** `text/event-stream`
- **Headers:** `Cache-Control: no-cache`, `X-Accel-Buffering: no`
- **Event Lifecycle:**
  1. `event: STARTED` — Run initialized.
  2. `event: DATASET_LOADING` — Documents, query groups, and relevance judgments loaded.
  3. `event: EMBEDDING_GENERATION` — Documents encoded into dense vectors.
  4. `event: RETRIEVAL_EXECUTION` — Vector candidate retrieval executed across queries.
  5. `event: METRIC_CALCULATION` — Precision, Recall, MRR, nDCG, bootstrap CIs, and paired Wilcoxon tests computed.
  6. `event: COMPLETED` (or `FAILED`) — Run finished and summary available.

#### `GET /api/v1/experiments/{experiment_id}/results`
Retrieve comprehensive evaluation aggregates, 95% bootstrap confidence intervals, paired Wilcoxon p-values, absolute performance gaps ($\Delta_{abs}$), relative disparity ratios ($D_{rel}$), and transliteration penalties ($\Phi_{trans}$).
- **Status:** `200 OK`
- **Response:**
  ```json
  {
    "experiment_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "experiment_name": "Cross-Lingual Disparity Study: Healthcare",
    "dataset_id": "2fa85f64-5717-4562-b3fc-2c963f66afa5",
    "baseline_language": "en",
    "runs": [
      {
        "id": "7fa85f64-5717-4562-b3fc-2c963f66afa7",
        "experiment_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "retrieval_mode": "EXACT",
        "status": "COMPLETED",
        "progress_percent": 100.0,
        "error_message": null,
        "device_used": "cpu",
        "hardware_metadata": {"cpu_threads": 8},
        "total_latency_seconds": 4.12,
        "created_at": "2026-11-10T02:00:00Z"
      }
    ],
    "metrics": [
      {
        "language_code": "en",
        "script_code": "Latn",
        "metric_name": "ndcg@10",
        "metric_value": 0.9412,
        "ci_lower_95": 0.9120,
        "ci_upper_95": 0.9701,
        "p_value_vs_baseline": null,
        "is_statistically_significant": false
      },
      {
        "language_code": "bn",
        "script_code": "Beng",
        "metric_name": "ndcg@10",
        "metric_value": 0.7230,
        "ci_lower_95": 0.6841,
        "ci_upper_95": 0.7618,
        "p_value_vs_baseline": 0.0034,
        "is_statistically_significant": true
      }
    ],
    "fairness_disparities": [
      {
        "language_code": "bn",
        "baseline_language": "en",
        "absolute_performance_gap": 0.2182,
        "relative_disparity_ratio": 0.7682,
        "p_value": 0.0034,
        "is_statistically_significant": true
      }
    ],
    "transliteration_penalties": {
      "bn_to_bn-Latn": 0.3120
    }
  }
  ```
