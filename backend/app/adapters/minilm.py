"""Adapter for paraphrase-multilingual-MiniLM-L12-v2.

A lightweight, 117M-parameter multilingual bi-encoder distilled from XLM-RoBERTa.
Accepts raw text with zero required task prefixes.
Uses native Hugging Face AutoModel + AutoTokenizer with Mean Pooling for maximum stability.
"""

from __future__ import annotations

import gc
from typing import Any, Sequence
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from app.adapters.base import BaseEmbeddingAdapter, ModelMetadata, TokenizationResult


class MiniLMEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2."""

    MODEL_IDENTIFIER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(model_id=self.MODEL_IDENTIFIER, device=device)
        self._model: Any = None
        self._tokenizer: Any = None

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.MODEL_IDENTIFIER,
            display_name="Paraphrase Multilingual MiniLM L12 v2",
            provider="Hugging Face / Sentence-Transformers",
            parameter_count=117_000_000,
            embedding_dimension=384,
            max_sequence_length=128,
            license="Apache-2.0",
            query_prefix="",
            document_prefix="",
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
                max_length=128,
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
        return self._encode_batch(queries, batch_size=batch_size, normalize=normalize)

    def encode_documents(
        self,
        documents: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        return self._encode_batch(documents, batch_size=batch_size, normalize=normalize)

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
