# LinguaLens: REST & Streaming API Specification

**Document Version:** 1.0.0  
**Base URL:** `/api/v1`  
**Protocol:** HTTP/1.1 & HTTP/2, Server-Sent Events (SSE) for streaming  
**Content-Type:** `application/json` (unless multipart for file uploads)  

---

## 1. Error Handling Standard

All non-2xx responses conform to a structured error envelope:

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested dataset 'bn-eval-v1' does not exist.",
    "details": {},
    "timestamp": "2026-11-09T15:30:00Z",
    "request_id": "req-9b814a70"
  }
}
```

---

## 2. API Endpoints

### 2.1 Authentication & Health

#### `GET /health`
Returns system status, active database dialect, and PyTorch execution device.
- **Response 200 OK:**
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "environment": "production",
    "device": "cpu",
    "cuda_available": false,
    "vector_backend": "numpy_in_memory"
  }
  ```

#### `POST /auth/register` & `POST /auth/login`
Standard JWT-based bearer authentication.

---

### 2.2 Datasets & Query Groups (`/datasets`)

#### `GET /datasets`
List all registered benchmark datasets with metadata and language counts.
- **Query Params:** `dataset_type` (`VALIDATED_BENCHMARK`, `DEMONSTRATION`), `limit`, `offset`.
- **Response 200 OK:**
  ```json
  {
    "total": 3,
    "items": [
      {
        "id": "c6a2b8e0-5a3d-4c3e-9b2f-7d1a5c6e8f01",
        "name": "Multilingual Intent Retrieval Benchmark",
        "slug": "mir-benchmark-v1",
        "version": "1.0.0",
        "dataset_type": "VALIDATED_BENCHMARK",
        "supported_languages": ["en", "bn", "bn-Latn", "hi", "ar"],
        "document_count": 5000,
        "query_group_count": 250,
        "query_count": 1250,
        "content_hash_sha256": "8f4e2c...",
        "created_at": "2026-11-01T12:00:00Z"
      }
    ]
  }
  ```

#### `POST /datasets/upload`
Upload a new dataset archive (JSONL or ZIP containing `queries.jsonl`, `documents.jsonl`, `qrels.jsonl`).
- **Content-Type:** `multipart/form-data`
- **Form Fields:** `name`, `version`, `dataset_type`, `file`
- **Validation:** Enforces maximum file size (100MB), valid schema structure, and ISO codes.

#### `GET /datasets/{id}/query-groups`
Retrieve paired query groups for inspection in the Query Lab.

---

### 2.3 Model Registry & Tokenizer Inspector (`/models`)

#### `GET /models`
List available embedding model adapters.
- **Response 200 OK:**
  ```json
  [
    {
      "id": "m1-minilm",
      "model_identifier": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "display_name": "Paraphrase Multilingual MiniLM L12 v2",
      "parameter_count": 117000000,
      "embedding_dimension": 384,
      "max_sequence_length": 512,
      "license": "Apache-2.0",
      "query_prefix": "",
      "supports_cross_encoder": false,
      "is_active": true
    },
    {
      "id": "m2-e5-small",
      "model_identifier": "intfloat/multilingual-e5-small",
      "display_name": "Multilingual E5 Small",
      "parameter_count": 118000000,
      "embedding_dimension": 384,
      "max_sequence_length": 512,
      "license": "MIT",
      "query_prefix": "query: ",
      "document_prefix": "passage: ",
      "supports_cross_encoder": false,
      "is_active": true
    }
  ]
  ```

#### `POST /models/tokenize-inspect`
Inspect how different tokenizers split multilingual and transliterated strings.
- **Request Body:**
  ```json
  {
    "model_id": "m2-e5-small",
    "text": "amar sonar bangla",
    "language_code": "bn-Latn"
  }
  ```
- **Response 200 OK:**
  ```json
  {
    "tokens": ["am", "##ar", "son", "##ar", "bang", "##la"],
    "token_ids": [1042, 2841, 9812, 2841, 14210, 3122],
    "token_count": 6,
    "has_unknown_tokens": false,
    "subword_fertility_ratio": 2.0
  }
  ```

---

### 2.4 Multilingual Query Lab (`/query-lab`)

#### `POST /query-lab/compare`
Executes interactive multi-lingual query search in parallel across indexed documents.
- **Request Body:**
  ```json
  {
    "dataset_id": "c6a2b8e0-5a3d-4c3e-9b2f-7d1a5c6e8f01",
    "model_id": "m2-e5-small",
    "queries": [
      {"language_code": "en", "query_text": "treatment for seasonal flu"},
      {"language_code": "bn", "query_text": "মৌসুমি ফ্লুর চিকিৎসা"},
      {"language_code": "bn-Latn", "query_text": "mousumi flur chikitsa"},
      {"language_code": "hi", "query_text": "मौसमी फ्लू का इलाज"},
      {"language_code": "ar", "query_text": "علاج الأنفلونزا الموسمية"}
    ],
    "top_k": 5
  }
  ```
- **Response 200 OK:**
  ```json
  {
    "model_identifier": "intfloat/multilingual-e5-small",
    "results_by_language": {
      "en": {
        "latency_ms": 14.2,
        "hits": [
          {"doc_id": "doc-042", "title": "Influenza Overview", "score": 0.892, "rank": 1}
        ]
      },
      "bn": {
        "latency_ms": 15.1,
        "hits": [
          {"doc_id": "doc-042", "title": "Influenza Overview", "score": 0.835, "rank": 1}
        ]
      },
      "bn-Latn": {
        "latency_ms": 14.8,
        "hits": [
          {"doc_id": "doc-199", "title": "General Wellness", "score": 0.612, "rank": 1}
        ]
      }
    },
    "jaccard_overlap_matrix": {
      "en_vs_bn": 0.80,
      "en_vs_bn-Latn": 0.20,
      "bn_vs_bn-Latn": 0.25
    }
  }
  ```

---

### 2.5 Experiments & Benchmarking (`/experiments`)

#### `POST /experiments`
Create and schedule a full evaluation experiment.
- **Request Body:**
  ```json
  {
    "name": "E5-Small vs MiniLM Cross-Lingual Evaluation",
    "dataset_id": "c6a2b8e0-5a3d-4c3e-9b2f-7d1a5c6e8f01",
    "model_ids": ["m1-minilm", "m2-e5-small"],
    "baseline_language": "en",
    "cutoff_k_values": [1, 5, 10, 20],
    "similarity_metric": "cosine",
    "compute_bootstrap_ci": true,
    "bootstrap_samples": 1000
  }
  ```
- **Response 202 Accepted:**
  ```json
  {
    "experiment_id": "e4f8-412a",
    "status": "QUEUED",
    "created_at": "2026-11-09T15:35:00Z",
    "runs": [
      {"run_id": "run-001", "model_id": "m1-minilm", "status": "QUEUED"},
      {"run_id": "run-002", "model_id": "m2-e5-small", "status": "QUEUED"}
    ]
  }
  ```

#### `GET /experiments/{id}/stream`
Server-Sent Events (SSE) streaming real-time experiment progress:
- **Event: `progress`**
  ```json
  {"run_id": "run-001", "stage": "ENCODING_QUERIES", "progress": 45.0, "current_language": "bn"}
  ```
- **Event: `completed`**
  ```json
  {"experiment_id": "e4f8-412a", "status": "COMPLETED", "summary_url": "/api/v1/experiments/e4f8-412a/results"}
  ```

#### `GET /experiments/{id}/results`
Returns comprehensive evaluation aggregates, disparity gaps, and confidence intervals.

---

### 2.6 Research Reports (`/reports`)

#### `POST /reports/generate`
Generates a formal research report from an experiment run.
- **Request Body:**
  ```json
  {
    "experiment_id": "e4f8-412a",
    "title": "Empirical Audit of Multilingual Search Disparity in South Asian Scripts",
    "format": "MARKDOWN"
  }
  ```
- **Response 201 Created:** Returns full markdown report with citation bibliography, statistical tables, and methodology disclaimer.
