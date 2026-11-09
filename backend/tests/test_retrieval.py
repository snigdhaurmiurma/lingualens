"""Integration tests for Exact and Vector-Index retrieval engines and adapter integration."""

import uuid
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.adapters.minilm import MiniLMEmbeddingAdapter
from app.db.session import Base
from app.retrieval.base import SearchResult
from app.retrieval.exact import NumPyExactVectorRetriever
from app.retrieval.pgvector import PgVectorRetriever
from app.services.dataset_service import DatasetService
from app.services.retrieval_service import RetrievalService


@pytest.fixture
def db_session() -> Session:
    """In-memory SQLite database session for retrieval tests."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_exact_retriever_analytical_cosine():
    """Verify exact cosine similarity calculation against analytical known vectors.

    Let:
      d1 = [1.0, 0.0, 0.0]  -> cos(q, d1) = 1.0
      d2 = [1/sqrt(2), 1/sqrt(2), 0.0] -> cos(q, d2) = 0.7071
      d3 = [0.0, 1.0, 0.0]  -> cos(q, d3) = 0.0
      d4 = [-1.0, 0.0, 0.0] -> cos(q, d4) = -1.0
      query = [1.0, 0.0, 0.0]
    """
    doc_ids = [uuid.uuid4() for _ in range(4)]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    retriever = NumPyExactVectorRetriever(
        document_ids=doc_ids,
        doc_external_ids=["d1", "d2", "d3", "d4"],
        embeddings=embeddings,
        contents=["doc 1", "doc 2", "doc 3", "doc 4"],
        titles=["T1", "T2", "T3", "T4"],
        languages=["en", "bn", "hi", "ar"],
        scripts=["Latn", "Beng", "Deva", "Arab"],
    )

    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = retriever.retrieve(query_vec, top_k=4)

    assert len(results) == 4
    # Rank 1: d1 with score 1.0
    assert results[0].doc_external_id == "d1"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[0].rank == 1

    # Rank 2: d2 with score ~0.707107
    assert results[1].doc_external_id == "d2"
    assert results[1].score == pytest.approx(1.0 / np.sqrt(2), abs=1e-5)
    assert results[1].rank == 2

    # Rank 3: d3 with score 0.0
    assert results[2].doc_external_id == "d3"
    assert results[2].score == pytest.approx(0.0, abs=1e-5)
    assert results[2].rank == 3

    # Rank 4: d4 with score -1.0
    assert results[3].doc_external_id == "d4"
    assert results[3].score == pytest.approx(-1.0, abs=1e-5)
    assert results[3].rank == 4


def test_exact_retriever_from_database(db_session: Session):
    """Verify loading from database and retrieving top-K documents."""
    ds = DatasetService.create_dataset(session=db_session, name="DB Retriever Test", slug="db-ret-test")

    # Add 3 documents
    d1 = DatasetService.add_document(session=db_session, dataset_id=ds.id, doc_external_id="d1", content="English medical doc", language_code="en", script_code="Latn")
    d2 = DatasetService.add_document(session=db_session, dataset_id=ds.id, doc_external_id="d2", content="Bengali medical doc", language_code="bn", script_code="Beng")
    d3 = DatasetService.add_document(session=db_session, dataset_id=ds.id, doc_external_id="d3", content="Paris distractor", language_code="en", script_code="Latn")

    # Fixed orthogonal/aligned 384d vectors
    v1 = np.zeros(384, dtype=np.float32); v1[0] = 1.0
    v2 = np.zeros(384, dtype=np.float32); v2[0] = 0.8; v2[1] = 0.6
    v3 = np.zeros(384, dtype=np.float32); v3[2] = 1.0  # orthogonal

    model_id = "test-model"
    DatasetService.store_document_embedding(session=db_session, document_id=d1.id, model_id=model_id, embedding=v1)
    DatasetService.store_document_embedding(session=db_session, document_id=d2.id, model_id=model_id, embedding=v2)
    DatasetService.store_document_embedding(session=db_session, document_id=d3.id, model_id=model_id, embedding=v3)

    retriever = NumPyExactVectorRetriever.from_database(session=db_session, model_id=model_id, dataset_id=ds.id)
    assert retriever.count() == 3

    q = np.zeros(384, dtype=np.float32); q[0] = 1.0
    results = retriever.retrieve(q, top_k=2)

    assert len(results) == 2
    assert results[0].doc_external_id == "d1"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[1].doc_external_id == "d2"
    assert results[1].score == pytest.approx(0.8, abs=1e-5)


def test_retrieval_mode_consistency(db_session: Session):
    """Verify that both Exact and Vector Index modes return identical result contracts."""
    ds = DatasetService.create_dataset(session=db_session, name="Consistency Test", slug="cons-test")
    doc = DatasetService.add_document(session=db_session, dataset_id=ds.id, doc_external_id="doc-1", content="Sample content", language_code="bn", script_code="Beng")

    vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
    model_id = "consistency-model"
    DatasetService.store_document_embedding(session=db_session, document_id=doc.id, model_id=model_id, embedding=vec)

    exact_retriever = NumPyExactVectorRetriever.from_database(session=db_session, model_id=model_id)
    pg_retriever = PgVectorRetriever(session=db_session, model_id=model_id, dimension=384)

    q = np.ones(384, dtype=np.float32) / np.sqrt(384)
    res_exact = exact_retriever.retrieve(q, top_k=1)
    res_index = pg_retriever.retrieve(q, top_k=1)

    assert len(res_exact) == 1
    assert len(res_index) == 1

    r_e = res_exact[0]
    r_i = res_index[0]

    assert r_e.document_id == r_i.document_id
    assert r_e.doc_external_id == r_i.doc_external_id
    assert r_e.score == pytest.approx(r_i.score, abs=1e-5)
    assert r_e.rank == r_i.rank == 1
    assert r_e.language_code == r_i.language_code == "bn"
    assert r_e.script_code == r_i.script_code == "Beng"


def test_language_filtering(db_session: Session):
    """Verify language filtering preserves target language documents strictly."""
    ds = DatasetService.create_dataset(session=db_session, name="Lang Filter Test", slug="lang-filter-test")

    d_en = DatasetService.add_document(session=db_session, dataset_id=ds.id, doc_external_id="en-1", content="English text", language_code="en", script_code="Latn")
    d_bn = DatasetService.add_document(session=db_session, dataset_id=ds.id, doc_external_id="bn-1", content="বাংলা টেক্সট", language_code="bn", script_code="Beng")
    d_ar = DatasetService.add_document(session=db_session, dataset_id=ds.id, doc_external_id="ar-1", content="نص عربي", language_code="ar", script_code="Arab")

    vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
    model_id = "lang-filter-model"
    for d in [d_en, d_bn, d_ar]:
        DatasetService.store_document_embedding(session=db_session, document_id=d.id, model_id=model_id, embedding=vec)

    retriever = NumPyExactVectorRetriever.from_database(session=db_session, model_id=model_id)

    q = np.ones(384, dtype=np.float32) / np.sqrt(384)
    bn_only_results = retriever.retrieve(q, top_k=10, filter_language="bn")

    assert len(bn_only_results) == 1
    assert bn_only_results[0].doc_external_id == "bn-1"
    assert bn_only_results[0].language_code == "bn"
    assert bn_only_results[0].script_code == "Beng"


def test_edge_cases_empty_and_zero_vector(db_session: Session):
    """Verify robust error handling on empty index, dimension mismatch, and zero query."""
    empty_retriever = NumPyExactVectorRetriever(
        document_ids=[],
        doc_external_ids=[],
        embeddings=np.empty((0, 384), dtype=np.float32),
        contents=[],
        titles=[],
        languages=[],
        scripts=[],
    )
    q = np.ones(384, dtype=np.float32)
    assert empty_retriever.retrieve(q, top_k=5) == []

    # Dimension mismatch
    with pytest.raises(ValueError, match="does not match"):
        populated = NumPyExactVectorRetriever(
            document_ids=[uuid.uuid4()],
            doc_external_ids=["d1"],
            embeddings=np.ones((1, 384), dtype=np.float32),
            contents=["c"],
            titles=[None],
            languages=["en"],
            scripts=["Latn"],
        )
        populated.retrieve(np.ones(128, dtype=np.float32))

    # Zero vector query
    with pytest.raises(ValueError, match="zero-vector"):
        populated.retrieve(np.zeros(384, dtype=np.float32))


def test_model_adapter_retrieval_integration(db_session: Session):
    """End-to-end integration test: index dataset with MiniLM and retrieve via RetrievalService."""
    adapter = MiniLMEmbeddingAdapter(device="cpu")
    adapter.load()

    ds = DatasetService.create_dataset(
        session=db_session,
        name="MiniLM Live Search Test",
        slug="minilm-live-test",
    )

    doc_texts = [
        ("doc-flu-en", "Influenza is a contagious respiratory illness caused by flu viruses.", "en", "Latn"),
        ("doc-flu-bn", "মৌসুমি ফ্লু একটি সংক্রামক শ্বাসযন্ত্রের রোগ যা ইনফ্লুয়েঞ্জা ভাইরাসের কারণে ঘটে।", "bn", "Beng"),
        ("doc-paris-en", "Paris is the capital of France and is famous for the Eiffel Tower.", "en", "Latn"),
    ]

    for ext_id, content, lang, script in doc_texts:
        DatasetService.add_document(
            session=db_session,
            dataset_id=ds.id,
            doc_external_id=ext_id,
            content=content,
            language_code=lang,
            script_code=script,
        )

    # Index dataset with MiniLM adapter
    num_indexed = DatasetService.index_dataset_with_model(
        session=db_session,
        dataset_id=ds.id,
        model_adapter=adapter,
        batch_size=3,
    )
    assert num_indexed == 3

    # Execute search for medical query
    results = RetrievalService.search(
        session=db_session,
        model_adapter=adapter,
        query_text="what are the symptoms of seasonal flu?",
        retrieval_mode="EXACT",
        top_k=2,
    )

    assert len(results) == 2
    # The English flu doc should rank first
    assert results[0].doc_external_id == "doc-flu-en"
    assert results[0].score > results[1].score

    # Compare retrieval modes
    comparison = RetrievalService.compare_retrieval_modes(
        session=db_session,
        model_adapter=adapter,
        query_text="what are the symptoms of seasonal flu?",
        top_k=2,
    )

    assert comparison["index_recall_at_k"] == 1.0
    assert comparison["overlap_count"] == 2

    adapter.unload()
