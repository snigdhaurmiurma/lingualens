"""Retrieval orchestration service executing exact or index-based vector searches."""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.adapters.base import BaseEmbeddingAdapter
from app.retrieval.base import SearchResult
from app.retrieval.exact import NumPyExactVectorRetriever
from app.retrieval.pgvector import PgVectorRetriever


class RetrievalService:
    """Orchestrates query encoding, retrieval dispatch, and mode comparison."""

    @staticmethod
    def search(
        session: Session,
        model_adapter: BaseEmbeddingAdapter,
        query_text: str,
        retrieval_mode: str = "EXACT",  # EXACT | VECTOR_INDEX
        top_k: int = 10,
        filter_language: str | None = None,
        filter_dataset_id: uuid.UUID | None = None,
    ) -> list[SearchResult]:
        """Encode query text using adapter and execute search in specified retrieval mode."""
        if not query_text.strip():
            return []

        # Encode query text with model adapter (handles task prefixes like 'query: ')
        query_vectors = model_adapter.encode_queries([query_text], normalize=True)
        query_vector = query_vectors[0]

        return RetrievalService.search_by_vector(
            session=session,
            model_id=model_adapter.model_id,
            query_vector=query_vector,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            filter_language=filter_language,
            filter_dataset_id=filter_dataset_id,
        )

    @staticmethod
    def search_by_vector(
        session: Session,
        model_id: str,
        query_vector: np.ndarray,
        retrieval_mode: str = "EXACT",
        top_k: int = 10,
        filter_language: str | None = None,
        filter_dataset_id: uuid.UUID | None = None,
    ) -> list[SearchResult]:
        """Dispatch query vector to exact or pgvector retriever."""
        mode_upper = retrieval_mode.upper()

        if mode_upper == "EXACT":
            retriever = NumPyExactVectorRetriever.from_database(
                session=session,
                model_id=model_id,
                dataset_id=filter_dataset_id,
            )
            return retriever.retrieve(
                query_vector=query_vector,
                top_k=top_k,
                filter_language=filter_language,
                filter_dataset_id=filter_dataset_id,
            )
        elif mode_upper in {"VECTOR_INDEX", "PGVECTOR"}:
            retriever = PgVectorRetriever(
                session=session,
                model_id=model_id,
                dimension=query_vector.shape[0],
            )
            return retriever.retrieve(
                query_vector=query_vector,
                top_k=top_k,
                filter_language=filter_language,
                filter_dataset_id=filter_dataset_id,
            )
        else:
            raise ValueError(
                f"Unknown retrieval mode '{retrieval_mode}'. Supported modes: 'EXACT', 'VECTOR_INDEX'"
            )

    @staticmethod
    def compare_retrieval_modes(
        session: Session,
        model_adapter: BaseEmbeddingAdapter,
        query_text: str,
        top_k: int = 10,
        filter_dataset_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """Fairly compare Exact search against Vector Index search for index fidelity."""
        query_vectors = model_adapter.encode_queries([query_text], normalize=True)
        query_vector = query_vectors[0]

        # Mode A: Exact Brute Force
        t0 = time.perf_counter()
        exact_results = RetrievalService.search_by_vector(
            session=session,
            model_id=model_adapter.model_id,
            query_vector=query_vector,
            retrieval_mode="EXACT",
            top_k=top_k,
            filter_dataset_id=filter_dataset_id,
        )
        exact_latency_ms = (time.perf_counter() - t0) * 1000

        # Mode B: Vector Index Search
        t1 = time.perf_counter()
        index_results = RetrievalService.search_by_vector(
            session=session,
            model_id=model_adapter.model_id,
            query_vector=query_vector,
            retrieval_mode="VECTOR_INDEX",
            top_k=top_k,
            filter_dataset_id=filter_dataset_id,
        )
        index_latency_ms = (time.perf_counter() - t1) * 1000

        exact_doc_ids = [r.document_id for r in exact_results]
        index_doc_ids = [r.document_id for r in index_results]

        # Compute Index Recall@K: fraction of exact top-K items captured by index
        overlap = set(exact_doc_ids).intersection(set(index_doc_ids))
        index_recall = len(overlap) / float(max(1, len(exact_doc_ids)))

        return {
            "query_text": query_text,
            "top_k": top_k,
            "exact_results": exact_results,
            "index_results": index_results,
            "exact_latency_ms": round(exact_latency_ms, 2),
            "index_latency_ms": round(index_latency_ms, 2),
            "index_recall_at_k": round(index_recall, 4),
            "exact_count": len(exact_results),
            "index_count": len(index_results),
            "overlap_count": len(overlap),
        }
