"""Semantic vector search API endpoint."""

from __future__ import annotations

import time
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.adapters.registry import default_registry
from app.db.session import get_db
from app.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/search", tags=["Semantic Search"])


@router.post("", response_model=SearchResponse, status_code=status.HTTP_200_OK)
def search_corpus(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    """Execute semantic vector search using model selection, top-K, and retrieval mode."""
    try:
        adapter = default_registry.get_adapter(payload.model_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Embedding model '{payload.model_id}' is not registered.",
        )

    t0 = time.perf_counter()
    try:
        results = RetrievalService.search(
            session=db,
            model_adapter=adapter,
            query_text=payload.query_text,
            retrieval_mode=payload.retrieval_mode,
            top_k=payload.top_k,
            filter_language=payload.filter_language,
            filter_dataset_id=payload.filter_dataset_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    items = [
        SearchResultItem(
            document_id=r.document_id,
            doc_external_id=r.doc_external_id,
            score=round(r.score, 6),
            rank=r.rank,
            language_code=r.language_code,
            script_code=r.script_code,
            title=r.title,
            content=r.content,
            metadata=r.metadata,
        )
        for r in results
    ]

    return SearchResponse(
        query_text=payload.query_text,
        model_id=adapter.model_id,
        retrieval_mode=payload.retrieval_mode,
        top_k=payload.top_k,
        count=len(items),
        latency_ms=round(latency_ms, 2),
        results=items,
    )
