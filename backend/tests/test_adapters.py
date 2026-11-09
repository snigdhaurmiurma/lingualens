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
