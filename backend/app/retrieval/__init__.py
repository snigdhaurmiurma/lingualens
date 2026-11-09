"""Vector retrieval module providing exact and approximate nearest-neighbor search."""

from app.retrieval.base import BaseVectorRetriever, SearchResult
from app.retrieval.exact import NumPyExactVectorRetriever
from app.retrieval.pgvector import PgVectorRetriever

__all__ = [
    "BaseVectorRetriever",
    "SearchResult",
    "NumPyExactVectorRetriever",
    "PgVectorRetriever",
]
