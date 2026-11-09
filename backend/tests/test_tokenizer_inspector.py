"""Unit tests for Tokenizer Inspector and Subword Diagnostics."""

import pytest
from transformers import AutoTokenizer

from app.adapters.e5 import MultilingualE5Adapter
from app.adapters.minilm import MiniLMEmbeddingAdapter
from app.services.tokenizer_service import TokenizerInspector


@pytest.fixture(scope="module")
def minilm_inspector():
    """Provides a TokenizerInspector using the local MiniLM tokenizer."""
    adapter = MiniLMEmbeddingAdapter()
    adapter._tokenizer = AutoTokenizer.from_pretrained(adapter.MODEL_IDENTIFIER)
    adapter._is_loaded = True
    return TokenizerInspector(adapter)


@pytest.fixture(scope="module")
def e5_inspector():
    """Provides a TokenizerInspector using the local E5-Small tokenizer."""
    adapter = MultilingualE5Adapter()
    adapter._tokenizer = AutoTokenizer.from_pretrained(adapter.MODEL_IDENTIFIER)
    adapter._is_loaded = True
    return TokenizerInspector(adapter)


def test_tokenizer_inspect_single_query(minilm_inspector):
    result = minilm_inspector.inspect_text("symptoms of seasonal flu", language_code="en")

    assert result.text == "symptoms of seasonal flu"
    assert result.language_code == "en"
    assert result.word_count == 4
    assert result.token_count > 0
    assert len(result.tokens) == result.token_count
    assert len(result.token_ids) == result.token_count
    assert result.subword_fertility_ratio >= 1.0


def test_multilingual_query_group_inspection(minilm_inspector):
    queries = {
        "en": "symptoms of seasonal flu",
        "bn": "মৌসুমি ফ্লুর লক্ষণ",
        "bn-Latn": "mousumi flur lokkhon",
        "hi": "मौसमी फ्लू के लक्षण",
        "ar": "أعراض الإنفلونزا الموسمية",
    }

    results = minilm_inspector.inspect_multilingual_query_group(queries)

    assert set(results.keys()) == {"en", "bn", "bn-Latn", "hi", "ar"}

    # Verify Bengali fragmentation
    assert results["bn"].word_count == 3
    assert results["bn"].token_count > results["bn"].word_count
    assert results["bn"].subword_fertility_ratio > 2.0

    # Verify Arabic fragmentation
    assert results["ar"].word_count == 3
    assert results["ar"].token_count > results["ar"].word_count
    assert results["ar"].subword_fertility_ratio > 2.0


def test_compare_fertility_relative_to_baseline(minilm_inspector):
    queries = {
        "en": "symptoms of seasonal flu",
        "bn": "মৌসুমি ফ্লুর লক্ষণ",
        "bn-Latn": "mousumi flur lokkhon",
    }

    fertility_diff = minilm_inspector.compare_fertility(queries, baseline_language="en")

    assert "en" in fertility_diff
    assert "bn" in fertility_diff
    assert "bn-Latn" in fertility_diff

    # Baseline language self-ratio must be exactly 1.0
    assert fertility_diff["en"] == 1.0
    # Bengali subword fertility must be higher than English
    assert fertility_diff["bn"] > 1.0


def test_e5_prefix_handling_in_adapter():
    adapter = MultilingualE5Adapter()
    assert adapter._format_input("seasonal flu", adapter.QUERY_PREFIX) == "query: seasonal flu"
    assert adapter._format_input("query: seasonal flu", adapter.QUERY_PREFIX) == "query: seasonal flu"
    assert adapter._format_input("influenza overview", adapter.DOCUMENT_PREFIX) == "passage: influenza overview"


def test_tokenizer_limitations_metadata():
    limitations = TokenizerInspector.get_methodological_limitations()
    assert "construct_boundary" in limitations
    assert "script_asymmetry" in limitations
    assert "transliteration_ambiguity" in limitations
