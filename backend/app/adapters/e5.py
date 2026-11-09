"""Adapter for intfloat/multilingual-e5-small.

A high-performance, 118M-parameter multilingual bi-encoder trained on 1B+ text pairs.
Strictly requires 'query: ' and 'passage: ' task prefixes for accurate retrieval.
"""

from __future__ import annotations

import gc
from typing import Any, Sequence
import numpy as np

from app.adapters.base import BaseEmbeddingAdapter, ModelMetadata, TokenizationResult


class MultilingualE5Adapter(BaseEmbeddingAdapter):
    """Adapter for intfloat/multilingual-e5-small."""

    MODEL_IDENTIFIER = "intfloat/multilingual-e5-small"
    QUERY_PREFIX = "query: "
    DOCUMENT_PREFIX = "passage: "

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(model_id=self.MODEL_IDENTIFIER, device=device)
        self._model: Any = None
        self._tokenizer: Any = None

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.MODEL_IDENTIFIER,
            display_name="Multilingual E5 Small",
            provider="Microsoft / intfloat",
            parameter_count=118_000_000,
            embedding_dimension=384,
            max_sequence_length=512,
            license="MIT",
            query_prefix=self.QUERY_PREFIX,
            document_prefix=self.DOCUMENT_PREFIX,
            supported_languages=["en", "bn", "bn-Latn", "hi", "ar"],
            requires_gpu=False,
        )

    def load(self) -> None:
        """Load model weights and tokenizer."""
        if self._is_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer

            self._model = SentenceTransformer(self.MODEL_IDENTIFIER, device=self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_IDENTIFIER)
            self._is_loaded = True
        except Exception as exc:
            self._is_loaded = False
            raise RuntimeError(
                f"Failed to load model '{self.MODEL_IDENTIFIER}': {exc}"
            ) from exc

    def unload(self) -> None:
        """Unload model from memory to free host RAM."""
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        gc.collect()

    def _format_input(self, text: str, prefix: str) -> str:
        """Prepend required task prefix if not already present."""
        if text.startswith(prefix):
            return text
        return f"{prefix}{text}"

    def encode_queries(
        self,
        queries: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        if not self._is_loaded:
            self.load()

        if not queries:
            return np.empty((0, 384), dtype=np.float32)

        prefixed_queries = [self._format_input(q, self.QUERY_PREFIX) for q in queries]
        embeddings = self._model.encode(
            prefixed_queries,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_documents(
        self,
        documents: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        if not self._is_loaded:
            self.load()

        if not documents:
            return np.empty((0, 384), dtype=np.float32)

        prefixed_docs = [self._format_input(d, self.DOCUMENT_PREFIX) for d in documents]
        embeddings = self._model.encode(
            prefixed_docs,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def tokenize(self, text: str, language_code: str = "unknown") -> TokenizationResult:
        if not self._is_loaded:
            self.load()

        encoded = self._tokenizer(text, add_special_tokens=True)
        token_ids: list[int] = encoded["input_ids"]
        tokens: list[str] = self._tokenizer.convert_ids_to_tokens(token_ids)

        words = text.strip().split()
        word_count = len(words)
        token_count = len(tokens)

        fertility_ratio = (
            token_count / max(1, word_count) if word_count > 0 else 0.0
        )

        unk_id = getattr(self._tokenizer, "unk_token_id", None)
        unk_count = token_ids.count(unk_id) if unk_id is not None else 0

        return TokenizationResult(
            text=text,
            language_code=language_code,
            tokens=tokens,
            token_ids=token_ids,
            token_count=token_count,
            word_count=word_count,
            subword_fertility_ratio=round(fertility_ratio, 2),
            has_unknown_tokens=(unk_count > 0),
            unknown_token_count=unk_count,
        )
