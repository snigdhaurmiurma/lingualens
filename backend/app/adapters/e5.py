"""Adapter for intfloat/multilingual-e5-small.

A high-performance, 118M-parameter multilingual bi-encoder trained on 1B+ text pairs.
Strictly requires 'query: ' and 'passage: ' task prefixes for accurate retrieval.
Uses native Hugging Face AutoModel + AutoTokenizer with Mean Pooling for maximum stability.
"""

from __future__ import annotations

import gc
from typing import Any, Sequence
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

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
        """Load model weights and tokenizer into memory."""
        if self._is_loaded:
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_IDENTIFIER)
            self._model = AutoModel.from_pretrained(self.MODEL_IDENTIFIER)
            self._model.to(self.device)
            self._model.eval()
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

    def _mean_pooling(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform standard mean pooling over token embeddings with attention masking."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode_batch(self, texts: Sequence[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        if not self._is_loaded:
            self.load()

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**encoded)
                pooled = self._mean_pooling(outputs, encoded["attention_mask"])
                if normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)

    def encode_queries(
        self,
        queries: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        prefixed_queries = [self._format_input(q, self.QUERY_PREFIX) for q in queries]
        return self._encode_batch(prefixed_queries, batch_size=batch_size, normalize=normalize)

    def encode_documents(
        self,
        documents: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        prefixed_docs = [self._format_input(d, self.DOCUMENT_PREFIX) for d in documents]
        return self._encode_batch(prefixed_docs, batch_size=batch_size, normalize=normalize)

    def tokenize(self, text: str, language_code: str = "unknown") -> TokenizationResult:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_IDENTIFIER)

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
