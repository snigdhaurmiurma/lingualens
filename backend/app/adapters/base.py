"""Abstract Base Embedding Adapter Interface.

Defines the contract for pluggable multilingual embedding models in LinguaLens.
Ensures strict model isolation, standardized batch inference, and prefix management.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing an embedding model's properties and resource requirements."""

    model_id: str
    display_name: str
    provider: str
    parameter_count: int
    embedding_dimension: int
    max_sequence_length: int
    license: str
    query_prefix: str = ""
    document_prefix: str = ""
    supported_languages: list[str] = field(default_factory=list)
    requires_gpu: bool = False


@dataclass(frozen=True)
class TokenizationResult:
    """Result of tokenizing a text string, including subword fragmentation statistics."""

    text: str
    language_code: str
    tokens: list[str]
    token_ids: list[int]
    token_count: int
    word_count: int
    subword_fertility_ratio: float
    has_unknown_tokens: bool
    unknown_token_count: int


class BaseEmbeddingAdapter(ABC):
    """Abstract base class for all embedding model adapters.

    Every concrete adapter wraps a specific model family (e.g., SentenceTransformers,
    Hugging Face AutoModel) and handles:
    1. Lazy loading and unloading of model weights to manage host RAM.
    2. Input prefix injection (e.g. 'query: ' for E5 models).
    3. Batch-wise vector inference with L2 normalization on CPU or GPU.
    4. Diagnostic tokenization inspection.
    """

    def __init__(self, model_id: str, device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Return True if model weights are currently loaded in memory."""
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """Load model weights and tokenizer into memory on the target device."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model weights and clear memory."""
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Return static metadata for this model."""
        pass

    @abstractmethod
    def encode_queries(
        self,
        queries: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode a batch of search queries into dense vector embeddings.

        Applies any model-specific query prefix (e.g., 'query: ') before encoding.

        Args:
            queries: Sequence of query strings.
            batch_size: Number of queries processed per batch on CPU.
            normalize: Whether to L2-normalize embeddings to unit length (norm = 1.0).

        Returns:
            2D numpy array of shape (len(queries), embedding_dimension) with float32 dtype.
        """
        pass

    @abstractmethod
    def encode_documents(
        self,
        documents: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode a batch of corpus documents/passages into dense vector embeddings.

        Applies any model-specific document prefix (e.g., 'passage: ') before encoding.

        Args:
            documents: Sequence of document or passage strings.
            batch_size: Number of documents processed per batch on CPU.
            normalize: Whether to L2-normalize embeddings to unit length.

        Returns:
            2D numpy array of shape (len(documents), embedding_dimension) with float32 dtype.
        """
        pass

    @abstractmethod
    def tokenize(self, text: str, language_code: str = "unknown") -> TokenizationResult:
        """Audit subword tokenization for the given text string.

        Args:
            text: Input string in native or transliterated script.
            language_code: Target language or script code (e.g., 'en', 'bn', 'bn-Latn').

        Returns:
            TokenizationResult with subword tokens, token IDs, and fertility ratio.
        """
        pass
