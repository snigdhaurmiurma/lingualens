"""PostgreSQL + pgvector live integration test suite.

This test suite verifies native PostgreSQL database execution with the pgvector extension,
including HNSW index creation and native cosine distance (<=>) retrieval.
When a live PostgreSQL instance with pgvector is not available in the current environment,
the tests are cleanly skipped with diagnostic instructions.
"""

from __future__ import annotations

import os
import uuid
import numpy as np
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from app.db.session import Base
from app.models.dataset import Dataset, Document
from app.models.embedding import DocumentEmbedding
from app.retrieval.pgvector import PgVectorRetriever
from app.services.dataset_service import DatasetService


def get_postgres_test_url() -> str:
    """Return the configured PostgreSQL test connection URL."""
    return os.getenv(
        "TEST_POSTGRES_URL",
        "postgresql://postgres:postgres@localhost:5432/lingualens_test",
    )


def check_postgres_pgvector_available() -> tuple[bool, str]:
    """Check whether a live PostgreSQL server with pgvector is accessible."""
    url = get_postgres_test_url()
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            # Check vector extension
            result = conn.execute(
                text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector';")
            ).scalar()
            if not result:
                return False, "PostgreSQL is reachable, but 'pgvector' extension is not available"
        return True, "PostgreSQL + pgvector is available"
    except Exception as exc:
        return False, f"PostgreSQL not reachable at {url}: {exc}"


POSTGRES_AVAILABLE, SKIP_REASON = check_postgres_pgvector_available()


@pytest.mark.skipif(not POSTGRES_AVAILABLE, reason=SKIP_REASON)
def test_native_postgres_pgvector_lifecycle():
    """Verify live PostgreSQL table creation, pgvector extension, HNSW indexing, and retrieval."""
    url = get_postgres_test_url()
    engine = create_engine(url)

    # 1. Enable extension
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    # 2. Create tables
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    try:
        # 3. Create HNSW index
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_test_doc_embeddings_hnsw
                    ON document_embeddings USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                    """
                )
            )
            conn.commit()

            # Confirm index exists in pg_indexes
            idx_exists = conn.execute(
                text(
                    "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_test_doc_embeddings_hnsw';"
                )
            ).scalar()
            assert idx_exists == 1

        # 4. Ingest synthetic documents and 384d unit vectors
        ds = DatasetService.create_dataset(
            session=session,
            name="Postgres Integration Benchmark",
            slug=f"pg-bench-{uuid.uuid4().hex[:8]}",
        )

        d1 = DatasetService.add_document(
            session=session,
            dataset_id=ds.id,
            doc_external_id="pg-d1",
            content="Medical influenza text",
            language_code="en",
            script_code="Latn",
        )
        d2 = DatasetService.add_document(
            session=session,
            dataset_id=ds.id,
            doc_external_id="pg-d2",
            content="Paris landmark text",
            language_code="en",
            script_code="Latn",
        )

        v1 = np.zeros(384, dtype=np.float32); v1[0] = 1.0  # Aligned with query
        v2 = np.zeros(384, dtype=np.float32); v2[1] = 1.0  # Orthogonal

        DatasetService.store_document_embedding(session, d1.id, "pg-model", v1)
        DatasetService.store_document_embedding(session, d2.id, "pg-model", v2)

        # 5. Execute native PgVectorRetriever
        retriever = PgVectorRetriever(session=session, model_id="pg-model", dimension=384)
        q = np.zeros(384, dtype=np.float32); q[0] = 1.0

        # Enable ef_search tuning for HNSW
        with engine.connect() as conn:
            conn.execute(text("SET hnsw.ef_search = 40;"))

        results = retriever.retrieve(q, top_k=2)

        assert len(results) == 2
        # d1 should match query with cosine similarity 1.0
        assert results[0].doc_external_id == "pg-d1"
        assert results[0].score == pytest.approx(1.0, abs=1e-4)
        assert results[0].rank == 1

        # d2 is orthogonal with cosine similarity 0.0
        assert results[1].doc_external_id == "pg-d2"
        assert results[1].score == pytest.approx(0.0, abs=1e-4)
        assert results[1].rank == 2

    finally:
        session.close()
        # Clean up tables
        Base.metadata.drop_all(engine)
