"""Unit tests for Embedding Model Adapters and Model Registry."""

import numpy as np
import pytest

from app.adapters.e5 import MultilingualE5Adapter
from app.adapters.minilm import MiniLMEmbeddingAdapter
from app.adapters.registry import ModelRegistry, default_registry


def test_model_registry_list_available_models():
    models = default_registry.list_available_models()
    assert len(models) >= 2

    model_ids = [m.model_id for m in models]
    assert MiniLMEmbeddingAdapter.MODEL_IDENTIFIER in model_ids
    assert MultilingualE5Adapter.MODEL_IDENTIFIER in model_ids

    for m in models:
        assert m.embedding_dimension == 384
        assert m.parameter_count > 100_000_000
        assert m.license in {"Apache-2.0", "MIT"}
        assert "bn" in m.supported_languages
        assert "ar" in m.supported_languages


def test_model_registry_aliases_and_lookup():
    registry = ModelRegistry()

    assert registry.is_registered("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    assert registry.is_registered("minilm")
    assert registry.is_registered("intfloat/multilingual-e5-small")
    assert registry.is_registered("e5-small")
    assert not registry.is_registered("non-existent-model")

    with pytest.raises(KeyError, match="not registered"):
        registry.get_adapter("non-existent-model")


def test_model_adapter_metadata_integrity():
    minilm = MiniLMEmbeddingAdapter()
    meta_minilm = minilm.get_metadata()
    assert meta_minilm.embedding_dimension == 384
    assert meta_minilm.query_prefix == ""
    assert meta_minilm.license == "Apache-2.0"

    e5 = MultilingualE5Adapter()
    meta_e5 = e5.get_metadata()
    assert meta_e5.embedding_dimension == 384
    assert meta_e5.query_prefix == "query: "
    assert meta_e5.document_prefix == "passage: "
    assert meta_e5.license == "MIT"


def test_empty_input_handling():
    minilm = MiniLMEmbeddingAdapter()
    minilm._is_loaded = True  # Mock load state for empty input test
    empty_queries = minilm.encode_queries([])
    empty_docs = minilm.encode_documents([])

    assert isinstance(empty_queries, np.ndarray)
    assert empty_queries.shape == (0, 384)
    assert isinstance(empty_docs, np.ndarray)
    assert empty_docs.shape == (0, 384)


def test_unload_lifecycle():
    adapter = MiniLMEmbeddingAdapter()
    adapter._is_loaded = True
    adapter._model = "mock_model"
    adapter._tokenizer = "mock_tokenizer"

    assert adapter.is_loaded is True
    adapter.unload()
    assert adapter.is_loaded is False
    assert adapter._model is None
    assert adapter._tokenizer is None


def test_real_cpu_inference_minilm():
    """Verify actual real CPU forward pass and L2 normalization with MiniLM."""
    adapter = MiniLMEmbeddingAdapter(device="cpu")
    adapter.load()
    assert adapter.is_loaded is True

    queries = ["symptoms of seasonal flu", "মৌসুমি ফ্লুর লক্ষণ"]
    embeddings = adapter.encode_queries(queries, batch_size=2, normalize=True)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32

    # L2 Norm check: unit length
    norms = np.linalg.norm(embeddings, axis=1)
    for norm_val in norms:
        assert norm_val == pytest.approx(1.0, rel=1e-3)

    # Identical text similarity check
    dup_embeddings = adapter.encode_queries(["test query", "test query"], normalize=True)
    sim = float(np.dot(dup_embeddings[0], dup_embeddings[1]))
    assert sim == pytest.approx(1.0, rel=1e-4)

    adapter.unload()
    assert adapter.is_loaded is False


def test_minilm_matches_official_sentence_transformers():
    """Verify custom MiniLM adapter reproduces official SentenceTransformer embeddings exactly."""
    from sentence_transformers import SentenceTransformer

    test_sentences = [
        "symptoms of seasonal flu",
        "মৌসুমি ফ্লুর লক্ষণ",
        "mousumi flur lokkhon",
        "मौसमी फ्लू के लक्षण",
        "أعراض الإنفلونزا الموسمية",
    ]

    # Official SentenceTransformer inference
    st_model = SentenceTransformer(MiniLMEmbeddingAdapter.MODEL_IDENTIFIER, device="cpu")
    st_embeddings = st_model.encode(test_sentences, normalize_embeddings=True)

    # Custom Adapter inference
    adapter = MiniLMEmbeddingAdapter(device="cpu")
    adapter.load()
    adapter_embeddings = adapter.encode_queries(test_sentences, normalize=True)

    # Validate shapes and exact numerical equivalence
    assert st_embeddings.shape == adapter_embeddings.shape
    max_diff = float(np.max(np.abs(st_embeddings - adapter_embeddings)))
    assert max_diff < 1e-4, f"Max difference between adapter and official implementation ({max_diff}) exceeded tolerance"

    adapter.unload()


def test_real_cpu_inference_e5():
    """Verify actual real CPU forward pass, asymmetric prefixes, and normalization with E5-Small."""
    adapter = MultilingualE5Adapter(device="cpu")
    adapter.load()
    assert adapter.is_loaded is True

    queries = ["symptoms of seasonal flu", "মৌসুমি ফ্লুর লক্ষণ"]
    documents = [
        "Influenza, commonly known as flu, is an infectious disease caused by influenza viruses.",
        "মৌসুমি ফ্লু একটি সংক্রামক রোগ যা ইনফ্লুয়েঞ্জা ভাইরাসের কারণে ঘটে।",
        "Paris is the capital and most populous city of France.",
    ]

    query_embeddings = adapter.encode_queries(queries, batch_size=2, normalize=True)
    doc_embeddings = adapter.encode_documents(documents, batch_size=3, normalize=True)

    assert isinstance(query_embeddings, np.ndarray)
    assert query_embeddings.shape == (2, 384)
    assert query_embeddings.dtype == np.float32

    assert isinstance(doc_embeddings, np.ndarray)
    assert doc_embeddings.shape == (3, 384)
    assert doc_embeddings.dtype == np.float32

    # L2 Norm check: unit length
    for norm_val in np.linalg.norm(query_embeddings, axis=1):
        assert norm_val == pytest.approx(1.0, rel=1e-3)

    for norm_val in np.linalg.norm(doc_embeddings, axis=1):
        assert norm_val == pytest.approx(1.0, rel=1e-3)

    # Retrieval sanity check: English medical query should be more similar to medical docs than to Paris doc
    sim_en_med = float(np.dot(query_embeddings[0], doc_embeddings[0]))
    sim_en_paris = float(np.dot(query_embeddings[0], doc_embeddings[2]))
    assert sim_en_med > sim_en_paris, f"Medical query ({sim_en_med}) should score higher than Paris distractor ({sim_en_paris})"

    adapter.unload()
    assert adapter.is_loaded is False
