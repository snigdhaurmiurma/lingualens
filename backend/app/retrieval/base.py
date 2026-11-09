"""Base contracts and data models for vector retrieval engines in LinguaLens."""

from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    """Standardized search result item returned across all retrieval modes."""

    document_id: uuid.UUID
    doc_external_id: str
    score: float
    rank: int  # 1-based rank: 1, 2, ..., K
    language_code: str
    script_code: str
    content: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "document_id": str(self.document_id),
            "doc_external_id": self.doc_external_id,
            "score": round(self.score, 6),
            "rank": self.rank,
            "language_code": self.language_code,
            "script_code": self.script_code,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
        }


class BaseVectorRetriever(abc.ABC):
    """Abstract interface for dense vector retrieval engines."""

    @abc.abstractmethod
    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_language: str | None = None,
        filter_dataset_id: uuid.UUID | None = None,
    ) -> list[SearchResult]:
        """Retrieve the top-K closest documents for a query vector.

        Args:
            query_vector: Dense 1D float32 numpy array representing the query.
            top_k: Number of nearest documents to retrieve.
            filter_language: Optional language code filter (e.g., 'bn', 'en').
            filter_dataset_id: Optional dataset UUID filter.

        Returns:
            Ranked list of SearchResult items ordered by relevance score descending.
        """
        pass

    @abc.abstractmethod
    def count(self) -> int:
        """Return the total number of indexed vectors."""
        pass
