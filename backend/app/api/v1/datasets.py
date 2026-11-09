"""Dataset and corpus management API endpoints."""

from __future__ import annotations

import uuid
from fastapi import APIRouter, Depends, HTTPException, Query as QueryParam, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.dataset import Dataset, Document, Query, QueryGroup
from app.schemas.dataset import (
    DatasetCreate,
    DatasetResponse,
    DocumentCreate,
    DocumentResponse,
    PaginatedDatasetResponse,
    QueryCreate,
    QueryGroupCreate,
    QueryGroupResponse,
    QueryResponse,
)
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(payload: DatasetCreate, db: Session = Depends(get_db)) -> DatasetResponse:
    """Create a new versioned evaluation dataset."""
    existing = db.execute(select(Dataset).where(Dataset.slug == payload.slug)).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset with slug '{payload.slug}' already exists.",
        )

    dataset = DatasetService.create_dataset(
        session=db,
        name=payload.name,
        slug=payload.slug,
        version=payload.version,
        description=payload.description,
        supported_languages=payload.supported_languages,
        dataset_type=payload.dataset_type,
        source_uri=payload.source_uri,
    )

    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        slug=dataset.slug,
        version=dataset.version,
        description=dataset.description,
        dataset_type=dataset.dataset_type,
        license=dataset.license,
        supported_languages=dataset.supported_languages,
        content_hash_sha256=dataset.content_hash_sha256,
        is_locked=dataset.is_locked,
        document_count=0,
        query_count=0,
        query_group_count=0,
        created_at=dataset.created_at,
    )


@router.get("", response_model=PaginatedDatasetResponse, status_code=status.HTTP_200_OK)
def list_datasets(
    limit: int = QueryParam(default=50, ge=1, le=100),
    offset: int = QueryParam(default=0, ge=0),
    db: Session = Depends(get_db),
) -> PaginatedDatasetResponse:
    """List datasets with pagination."""
    total = db.scalar(select(func.count()).select_from(Dataset)) or 0
    datasets = list(
        db.execute(select(Dataset).order_by(Dataset.created_at.desc()).limit(limit).offset(offset))
        .scalars()
        .all()
    )

    responses: list[DatasetResponse] = []
    for d in datasets:
        doc_count = len(d.documents)
        query_count = len(d.queries)
        group_count = len(d.query_groups)

        responses.append(
            DatasetResponse(
                id=d.id,
                name=d.name,
                slug=d.slug,
                version=d.version,
                description=d.description,
                dataset_type=d.dataset_type,
                license=d.license,
                supported_languages=d.supported_languages,
                content_hash_sha256=d.content_hash_sha256,
                is_locked=d.is_locked,
                document_count=doc_count,
                query_count=query_count,
                query_group_count=group_count,
                created_at=d.created_at,
            )
        )
    return PaginatedDatasetResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=responses,
    )


@router.get("/{dataset_id}", response_model=DatasetResponse, status_code=status.HTTP_200_OK)
def get_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> DatasetResponse:
    """Retrieve details of a dataset by ID."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found.",
        )

    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        slug=dataset.slug,
        version=dataset.version,
        description=dataset.description,
        dataset_type=dataset.dataset_type,
        license=dataset.license,
        supported_languages=dataset.supported_languages,
        content_hash_sha256=dataset.content_hash_sha256,
        is_locked=dataset.is_locked,
        document_count=len(dataset.documents),
        query_count=len(dataset.queries),
        query_group_count=len(dataset.query_groups),
        created_at=dataset.created_at,
    )


@router.post("/{dataset_id}/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def add_document_to_dataset(
    dataset_id: uuid.UUID,
    payload: DocumentCreate,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """Add a passage/document to a dataset."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found.",
        )

    doc = DatasetService.add_document(
        session=db,
        dataset_id=dataset_id,
        doc_external_id=payload.doc_external_id,
        content=payload.content,
        language_code=payload.language_code,
        script_code=payload.script_code,
        title=payload.title,
        metadata=payload.metadata,
    )

    return DocumentResponse(
        id=doc.id,
        dataset_id=doc.dataset_id,
        doc_external_id=doc.doc_external_id,
        title=doc.title,
        content=doc.content,
        language_code=doc.language_code,
        script_code=doc.script_code,
        metadata=doc.metadata_json,
        created_at=doc.created_at,
    )


@router.post("/{dataset_id}/query-groups", response_model=QueryGroupResponse, status_code=status.HTTP_201_CREATED)
def add_query_group_to_dataset(
    dataset_id: uuid.UUID,
    payload: QueryGroupCreate,
    db: Session = Depends(get_db),
) -> QueryGroupResponse:
    """Add an invariant semantic query group."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found.",
        )

    group = DatasetService.add_query_group(
        session=db,
        dataset_id=dataset_id,
        canonical_intent=payload.canonical_intent,
        domain=payload.domain,
    )

    return QueryGroupResponse(
        id=group.id,
        dataset_id=group.dataset_id,
        canonical_intent=group.canonical_intent,
        domain=group.domain,
        created_at=group.created_at,
    )


@router.post("/{dataset_id}/queries", response_model=QueryResponse, status_code=status.HTTP_201_CREATED)
def add_query_to_dataset(
    dataset_id: uuid.UUID,
    payload: QueryCreate,
    db: Session = Depends(get_db),
) -> QueryResponse:
    """Add a localized query string realization."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found.",
        )

    group = db.get(QueryGroup, payload.query_group_id)
    if not group or group.dataset_id != dataset_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"QueryGroup '{payload.query_group_id}' does not belong to dataset '{dataset_id}'.",
        )

    q = DatasetService.add_query(
        session=db,
        query_group_id=payload.query_group_id,
        dataset_id=dataset_id,
        query_text=payload.query_text,
        language_code=payload.language_code,
        script_code=payload.script_code,
        metadata=payload.metadata,
    )

    return QueryResponse(
        id=q.id,
        query_group_id=q.query_group_id,
        dataset_id=q.dataset_id,
        query_text=q.query_text,
        language_code=q.language_code,
        script_code=q.script_code,
        metadata=q.metadata_json,
        created_at=q.created_at,
    )
