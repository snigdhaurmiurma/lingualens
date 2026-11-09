"""Integration tests for FastAPI REST & Streaming Gateway endpoints."""

import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db
from app.main import app
from app.services.dataset_service import DatasetService


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    """Provide a TestClient with an in-memory SQLite database using StaticPool."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_and_system_info(client: TestClient):
    """Verify health, system environment, and model catalog endpoints."""
    # 1. Health
    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200
    data_h = res_health.json()
    assert data_h["status"] == "healthy"
    assert data_h["database"] == "connected"

    # 2. System info
    res_info = client.get("/api/v1/system/info")
    assert res_info.status_code == 200
    data_i = res_info.json()
    assert "torch_version" in data_i
    assert len(data_i["registered_models"]) >= 2

    # 3. Model catalog
    res_models = client.get("/api/v1/models")
    assert res_models.status_code == 200
    data_m = res_models.json()
    assert len(data_m) >= 2
    model_ids = [m["model_id"] for m in data_m]
    assert "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" in model_ids
    assert "intfloat/multilingual-e5-small" in model_ids


def test_datasets_crud_operations(client: TestClient):
    """Verify dataset creation, listing, document ingestion, and query addition."""
    # 1. Create dataset
    ds_payload = {
        "name": "Clinical NLP Multilingual",
        "slug": "clinical-nlp-multi",
        "version": "1.0.0",
        "description": "Multi-dialect clinical dataset",
        "supported_languages": ["en", "bn", "hi"],
        "dataset_type": "VALIDATED_BENCHMARK",
    }
    res_create = client.post("/api/v1/datasets", json=ds_payload)
    assert res_create.status_code == 201
    dataset_data = res_create.json()
    ds_id = dataset_data["id"]
    assert dataset_data["slug"] == "clinical-nlp-multi"

    # 2. Duplicate slug should return 409
    res_dup = client.post("/api/v1/datasets", json=ds_payload)
    assert res_dup.status_code == 409

    # 3. List datasets
    res_list = client.get("/api/v1/datasets")
    assert res_list.status_code == 200
    data_list = res_list.json()
    assert "items" in data_list
    assert data_list["total"] >= 1
    assert len(data_list["items"]) >= 1

    # 4. Get dataset by ID
    res_get = client.get(f"/api/v1/datasets/{ds_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == ds_id

    # 5. Add document
    doc_payload = {
        "doc_external_id": "doc-01",
        "title": "Influenza Clinical Guide",
        "content": "Symptoms include high fever and cough.",
        "language_code": "en",
        "script_code": "Latn",
        "metadata": {"source": "CDC"},
    }
    res_doc = client.post(f"/api/v1/datasets/{ds_id}/documents", json=doc_payload)
    assert res_doc.status_code == 201
    assert res_doc.json()["doc_external_id"] == "doc-01"

    # 6. Add query group and query
    qg_payload = {
        "canonical_intent": "What are flu symptoms?",
        "domain": "healthcare",
    }
    res_qg = client.post(f"/api/v1/datasets/{ds_id}/query-groups", json=qg_payload)
    assert res_qg.status_code == 201
    qg_id = res_qg.json()["id"]

    q_payload = {
        "query_group_id": qg_id,
        "query_text": "flu symptoms",
        "language_code": "en",
        "script_code": "Latn",
    }
    res_q = client.post(f"/api/v1/datasets/{ds_id}/queries", json=q_payload)
    assert res_q.status_code == 201
    assert res_q.json()["query_text"] == "flu symptoms"


def test_tokenization_inspect_endpoint(client: TestClient):
    """Verify tokenization inspector endpoint calculation and limitations note."""
    payload = {
        "text": "মৌসুমি ফ্লুর লক্ষণ",
        "model_id": "minilm",
        "language_code": "bn",
    }
    res = client.post("/api/v1/tokenization/inspect", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["language_code"] == "bn"
    assert data["word_count"] == 3
    assert data["subword_fertility_ratio"] == 2.67
    assert len(data["text_tokens"]) == 8
    assert "Subword fertility measures vocabulary fragmentation" in data["limitations_note"]


def test_semantic_search_endpoint(client: TestClient):
    """Verify POST /api/v1/search with model selection, top-k, and retrieval modes."""
    # Create dataset and add documents
    res_ds = client.post(
        "/api/v1/datasets",
        json={"name": "Search Test DS", "slug": "search-test-ds", "supported_languages": ["en", "bn"]},
    )
    ds_id = res_ds.json()["id"]

    # Ingest document
    client.post(
        f"/api/v1/datasets/{ds_id}/documents",
        json={
            "doc_external_id": "doc-search-1",
            "content": "Influenza causes fever, cough, and fatigue.",
            "language_code": "en",
            "script_code": "Latn",
        },
    )

    # Search with minilm
    search_payload = {
        "query_text": "fever and fatigue causes",
        "model_id": "minilm",
        "retrieval_mode": "EXACT",
        "top_k": 5,
    }
    res_search = client.post("/api/v1/search", json=search_payload)
    assert res_search.status_code == 200
    s_data = res_search.json()
    assert s_data["query_text"] == "fever and fatigue causes"
    assert s_data["retrieval_mode"] == "EXACT"
    assert "latency_ms" in s_data


def test_invalid_requests_and_error_handling(client: TestClient):
    """Verify centralized validation and clean error sanitization."""
    # 1. Invalid UUID
    res_invalid_uuid = client.get("/api/v1/datasets/not-a-valid-uuid")
    assert res_invalid_uuid.status_code == 422
    err_data = res_invalid_uuid.json()
    assert err_data["error"] is True

    # 2. Non-existent dataset
    random_uuid = str(uuid.uuid4())
    res_not_found = client.get(f"/api/v1/datasets/{random_uuid}")
    assert res_not_found.status_code == 404

    # 3. Unknown model ID
    res_bad_model = client.post(
        "/api/v1/search",
        json={"query_text": "test", "model_id": "non-existent-model"},
    )
    assert res_bad_model.status_code == 404


def test_experiment_lifecycle_and_evaluation_results(client: TestClient):
    """Verify experiment configuration, run trigger, and results summary."""
    # 1. Setup dataset with documents and queries
    res_ds = client.post(
        "/api/v1/datasets",
        json={"name": "Exp Benchmark", "slug": "exp-benchmark"},
    )
    ds_id = res_ds.json()["id"]

    d1 = client.post(
        f"/api/v1/datasets/{ds_id}/documents",
        json={"doc_external_id": "d1", "content": "Flu symptoms overview", "language_code": "en", "script_code": "Latn"},
    ).json()["id"]

    qg = client.post(
        f"/api/v1/datasets/{ds_id}/query-groups",
        json={"canonical_intent": "Flu symptoms", "domain": "healthcare"},
    ).json()["id"]

    q1 = client.post(
        f"/api/v1/datasets/{ds_id}/queries",
        json={"query_group_id": qg, "query_text": "flu symptoms", "language_code": "en", "script_code": "Latn"},
    ).json()["id"]

    # 2. Create experiment
    exp_payload = {
        "name": "Multilingual Benchmark Study 1",
        "description": "Baseline MiniLM comparison",
        "dataset_id": ds_id,
        "baseline_language_code": "en",
        "cutoff_k_values": [1, 5, 10],
    }
    res_exp = client.post("/api/v1/experiments", json=exp_payload)
    assert res_exp.status_code == 201
    exp_id = res_exp.json()["id"]

    # 3. Trigger run
    run_payload = {
        "model_id": "minilm",
        "retrieval_mode": "EXACT",
    }
    res_run = client.post(f"/api/v1/experiments/{exp_id}/run", json=run_payload)
    assert res_run.status_code == 202
    run_data = res_run.json()
    assert run_data["status"] == "PENDING"

    # 4. Results endpoint returns summary
    res_results = client.get(f"/api/v1/experiments/{exp_id}/results")
    assert res_results.status_code == 200
    res_data = res_results.json()
    assert res_data["experiment_id"] == exp_id
    assert res_data["baseline_language"] == "en"


def test_sse_streaming_endpoint(client: TestClient):
    """Verify Server-Sent Events (SSE) streaming endpoint headers and handshake."""
    test_exp_id = str(uuid.uuid4())
    # Connect to stream
    with client.stream("GET", f"/api/v1/experiments/{test_exp_id}/stream") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert response.headers["cache-control"] == "no-cache"

        # Read first line
        lines = []
        for line in response.iter_lines():
            if line:
                lines.append(line)
            if len(lines) >= 2:
                break

        assert any("event: CONNECTED" in l for l in lines)


def test_cors_headers(client: TestClient):
    """Verify CORS middleware headers for frontend origin."""
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ["http://localhost:3000", "*"]


def test_pagination_and_invalid_parameters(client: TestClient):
    """Verify limit/offset pagination defaults, boundaries, and validation errors."""
    # 1. Valid pagination with metadata
    res = client.get("/api/v1/datasets?limit=10&offset=0")
    assert res.status_code == 200
    body = res.json()
    assert body["limit"] == 10
    assert body["offset"] == 0
    assert "total" in body
    assert isinstance(body["items"], list)

    # 2. Invalid limit < 1 should return 422
    res_zero = client.get("/api/v1/datasets?limit=0")
    assert res_zero.status_code == 422
    assert res_zero.json()["error"] is True

    # 3. Invalid limit > 100 should return 422
    res_large = client.get("/api/v1/datasets?limit=150")
    assert res_large.status_code == 422

    # 4. Invalid offset < 0 should return 422
    res_neg = client.get("/api/v1/datasets?offset=-5")
    assert res_neg.status_code == 422

    # 5. Experiments pagination validation
    res_exp = client.get("/api/v1/experiments?limit=5&offset=0")
    assert res_exp.status_code == 200
    exp_body = res_exp.json()
    assert exp_body["limit"] == 5
    assert exp_body["offset"] == 0
    assert "total" in exp_body

    res_exp_bad = client.get("/api/v1/experiments?limit=-1")
    assert res_exp_bad.status_code == 422


@pytest.mark.asyncio
async def test_sse_replay_for_late_joining_clients():
    """Verify that late-joining SSE subscribers receive historical events."""
    from app.services.event_stream import stream_manager
    test_id = uuid.uuid4()

    # Broadcast events before any client subscribes
    await stream_manager.broadcast(
        experiment_id=test_id,
        event_type="STARTED",
        data={"run_id": "r1", "progress": 0.0},
    )
    await stream_manager.broadcast(
        experiment_id=test_id,
        event_type="DATASET_LOADING",
        data={"run_id": "r1", "progress": 15.0},
    )

    # Client connects later
    queue = stream_manager.subscribe(test_id, replay_history=True)
    try:
        assert queue.qsize() == 2
        e1 = queue.get_nowait()
        assert "event: STARTED" in e1
        assert '"progress": 0.0' in e1

        e2 = queue.get_nowait()
        assert "event: DATASET_LOADING" in e2
        assert '"progress": 15.0' in e2
    finally:
        stream_manager.unsubscribe(test_id, queue)
        stream_manager.clear_history(test_id)


def test_postgresql_no_silent_fallback_to_sqlite():
    """Verify that configuring a PostgreSQL URL does not silently fall back to SQLite."""
    from app.db.session import get_engine

    unreachable_pg_url = "postgresql+psycopg2://invalid_user:invalid_pass@127.0.0.1:54399/nonexistent_db"
    
    # Must fail with PostgreSQL DBAPI or connection error; must NEVER silently fallback to SQLite
    with pytest.raises((ModuleNotFoundError, Exception)) as exc_info:
        pg_engine = get_engine(database_url=unreachable_pg_url)
        with pg_engine.connect():
            pass

    # Ensure the exception is postgres-specific and does not contain SQLite references
    assert "sqlite" not in str(exc_info.value).lower()

