"""Vector-index based retrieval engine integrating PostgreSQL pgvector."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.dataset import Document
from app.models.embedding import DocumentEmbedding
from app.retrieval.base import BaseVectorRetriever, SearchResult


class PgVectorRetriever(BaseVectorRetriever):
    """Vector-index retrieval engine leveraging PostgreSQL + pgvector.

    In production/PostgreSQL environments, uses the `<=>` cosine distance operator
    with HNSW or IVFFlat index acceleration.
    In local SQLite testing environments, provides transparent fallback emulation
    to verify database-backed query pipelines without external service requirements.
    """

    def __init__(
        self,
        session: Session,
        model_id: str,
        dimension: int = 384,
    ) -> None:
        """Initialize the pgvector retrieval engine.

        Args:
            session: Active SQLAlchemy database session.
            model_id: Embedding model identifier (e.g. 'intfloat/multilingual-e5-small').
            dimension: Expected embedding dimension (default 384).
        """
        self.session = session
        self.model_id = model_id
        self.dimension = dimension

    def count(self) -> int:
        """Count the number of indexed vectors for this model."""
        stmt = (
            select(DocumentEmbedding.id)
            .where(DocumentEmbedding.model_id == self.model_id)
        )
        return len(self.session.execute(stmt).all())

    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_language: str | None = None,
        filter_dataset_id: uuid.UUID | None = None,
    ) -> list[SearchResult]:
        """Execute vector similarity search using pgvector or dialect-aware fallback."""
        if top_k <= 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dimension:
            raise ValueError(
                f"Query vector dimension ({q.shape[0]}) does not match model dimension ({self.dimension})"
            )

        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            raise ValueError("Cannot search with a zero-vector query.")
        q = q / q_norm

        dialect_name = self.session.bind.dialect.name if self.session.bind else "sqlite"

        if dialect_name == "postgresql":
            return self._retrieve_postgresql(
                query_vector=q,
                top_k=top_k,
                filter_language=filter_language,
                filter_dataset_id=filter_dataset_id,
            )
        else:
            return self._retrieve_sqlite_fallback(
                query_vector=q,
                top_k=top_k,
                filter_language=filter_language,
                filter_dataset_id=filter_dataset_id,
            )

    def _retrieve_postgresql(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filter_language: str | None,
        filter_dataset_id: uuid.UUID | None,
    ) -> list[SearchResult]:
        """Native PostgreSQL pgvector search using cosine distance operator `<=>`."""
        # Using pgvector cosine distance: score = 1 - distance
        dist_expr = DocumentEmbedding.embedding.cosine_distance(query_vector.tolist())

        stmt = (
            select(
                Document.id,
                Document.doc_external_id,
                Document.title,
                Document.content,
                Document.language_code,
                Document.script_code,
                Document.metadata_json,
                dist_expr.label("distance"),
            )
            .join(DocumentEmbedding, Document.id == DocumentEmbedding.document_id)
            .where(DocumentEmbedding.model_id == self.model_id)
        )

        if filter_language is not None:
            stmt = stmt.where(Document.language_code == filter_language)
        if filter_dataset_id is not None:
            stmt = stmt.where(Document.dataset_id == filter_dataset_id)

        stmt = stmt.order_by(dist_expr.asc()).limit(top_k)
        rows = self.session.execute(stmt).all()

        results: list[SearchResult] = []
        for rank, row in enumerate(rows, start=1):
            distance = float(row[7])
            # Convert cosine distance to cosine similarity: sim = 1.0 - distance
            similarity = float(np.clip(1.0 - distance, -1.0, 1.0))

            results.append(
                SearchResult(
                    document_id=row[0],
                    doc_external_id=row[1],
                    score=similarity,
                    rank=rank,
                    language_code=row[4],
                    script_code=row[5],
                    title=row[2],
                    content=row[3],
                    metadata=row[6] or {},
                )
            )

        return results

    def _retrieve_sqlite_fallback(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filter_language: str | None,
        filter_dataset_id: uuid.UUID | None,
    ) -> list[SearchResult]:
        """Dialect fallback for local development / testing without a running PostgreSQL server."""
        from app.retrieval.exact import NumPyExactVectorRetriever

        exact_retriever = NumPyExactVectorRetriever.from_database(
            session=self.session,
            model_id=self.model_id,
            dataset_id=filter_dataset_id,
        )
        return exact_retriever.retrieve(
            query_vector=query_vector,
            top_k=top_k,
            filter_language=filter_language,
            filter_dataset_id=filter_dataset_id,
        )
