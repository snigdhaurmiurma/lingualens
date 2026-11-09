"""Tests for database persistence, relationships, constraints, and embedding storage."""

import uuid
import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.dataset import Dataset, Document, Query, QueryGroup, RelevanceJudgment
from app.models.embedding import DocumentEmbedding
from app.models.experiment import EvaluationResult, Experiment, ExperimentRun, RetrievedDocument
from app.services.dataset_service import DatasetService


@pytest.fixture
def db_session() -> Session:
    """Provide a fresh in-memory SQLite database session for each test."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_database_connection_and_table_creation(db_session: Session):
    """Verify that all tables exist and can be queried."""
    result = db_session.execute(select(Dataset)).all()
    assert result == []


def test_dataset_lifecycle_and_constraints(db_session: Session):
    """Verify dataset creation, unique constraints, and hash computation."""
    ds = DatasetService.create_dataset(
        session=db_session,
        name="Medical Multilingual Benchmark",
        slug="med-multi-bench",
        version="1.0.0",
        description="Clinical symptoms across 5 languages",
        supported_languages=["en", "bn", "hi", "ar"],
    )

    assert ds.id is not None
    assert ds.name == "Medical Multilingual Benchmark"
    assert len(ds.content_hash_sha256) == 64
    assert ds.supported_languages == ["en", "bn", "hi", "ar"]

    # Verify unique slug constraint
    with pytest.raises(IntegrityError):
        DatasetService.create_dataset(
            session=db_session,
            name="Another Dataset",
            slug="med-multi-bench",  # Duplicate slug
            version="1.0.0",
        )
    db_session.rollback()


def test_query_group_and_multilingual_queries(db_session: Session):
    """Verify query group hierarchy and language/script metadata preservation."""
    ds = DatasetService.create_dataset(
        session=db_session,
        name="Multilingual Query Test",
        slug="multi-q-test",
    )

    group = DatasetService.add_query_group(
        session=db_session,
        dataset_id=ds.id,
        canonical_intent="Symptoms of seasonal flu",
        domain="healthcare",
    )

    q_en = DatasetService.add_query(
        session=db_session,
        query_group_id=group.id,
        dataset_id=ds.id,
        query_text="symptoms of seasonal flu",
        language_code="en",
        script_code="Latn",
        metadata={"source": "who"},
    )

    q_bn = DatasetService.add_query(
        session=db_session,
        query_group_id=group.id,
        dataset_id=ds.id,
        query_text="মৌসুমি ফ্লুর লক্ষণ",
        language_code="bn",
        script_code="Beng",
        metadata={"transliteration": "mousumi flur lokkhon"},
    )

    assert q_en.query_group_id == group.id
    assert q_bn.query_group_id == group.id
    assert q_bn.script_code == "Beng"
    assert q_bn.metadata_json["transliteration"] == "mousumi flur lokkhon"

    # Verify cascading delete
    db_session.delete(group)
    db_session.commit()

    remaining_queries = db_session.execute(select(Query)).scalars().all()
    assert remaining_queries == []


def test_document_and_embedding_persistence(db_session: Session):
    """Verify storing high-dimensional vectors and retrieving as numpy arrays."""
    ds = DatasetService.create_dataset(
        session=db_session,
        name="Embedding Storage Test",
        slug="emb-store-test",
    )

    doc = DatasetService.add_document(
        session=db_session,
        dataset_id=ds.id,
        doc_external_id="doc-001",
        title="Flu Overview",
        content="Influenza is a contagious respiratory illness caused by influenza viruses.",
        language_code="en",
        script_code="Latn",
    )

    # Synthetic 384-dimensional unit vector
    rng = np.random.default_rng(42)
    raw_vec = rng.normal(0, 1, 384).astype(np.float32)
    unit_vec = raw_vec / np.linalg.norm(raw_vec)

    emb_row = DatasetService.store_document_embedding(
        session=db_session,
        document_id=doc.id,
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embedding=unit_vec,
    )

    assert emb_row.id is not None
    assert emb_row.dimension == 384

    # Query back from DB
    loaded_emb = db_session.execute(
        select(DocumentEmbedding).where(DocumentEmbedding.id == emb_row.id)
    ).scalar_one()

    assert isinstance(loaded_emb.embedding, np.ndarray)
    assert loaded_emb.embedding.shape == (384,)
    assert loaded_emb.embedding.dtype == np.float32
    assert np.allclose(loaded_emb.embedding, unit_vec, atol=1e-6)


def test_relevance_judgment_bounds_and_uniqueness(db_session: Session):
    """Verify graded relevance check constraints (0 <= r <= 3) and uniqueness."""
    ds = DatasetService.create_dataset(session=db_session, name="Rel Test", slug="rel-test")
    group = DatasetService.add_query_group(session=db_session, dataset_id=ds.id, canonical_intent="Intent")
    q = DatasetService.add_query(session=db_session, query_group_id=group.id, dataset_id=ds.id, query_text="query", language_code="en", script_code="Latn")
    d = DatasetService.add_document(session=db_session, dataset_id=ds.id, doc_external_id="d1", content="content", language_code="en", script_code="Latn")

    # Valid rating 3
    rj = DatasetService.add_relevance_judgment(session=db_session, dataset_id=ds.id, query_id=q.id, document_id=d.id, relevance_score=3)
    assert rj.relevance_score == 3

    # Invalid rating > 3 should fail validation
    with pytest.raises(ValueError, match="between 0 and 3"):
        DatasetService.add_relevance_judgment(session=db_session, dataset_id=ds.id, query_id=q.id, document_id=d.id, relevance_score=5)

    # Duplicate query-doc judgment should violate unique constraint
    with pytest.raises(IntegrityError):
        DatasetService.add_relevance_judgment(session=db_session, dataset_id=ds.id, query_id=q.id, document_id=d.id, relevance_score=2)
    db_session.rollback()
