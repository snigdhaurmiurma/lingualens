"""Dataset management service for creating datasets, indexing documents, and storing embeddings."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.base import BaseEmbeddingAdapter
from app.models.dataset import Dataset, Document, Query, QueryGroup, RelevanceJudgment
from app.models.embedding import DocumentEmbedding


class DatasetService:
    """Service handling dataset lifecycles, document ingestion, and embedding persistence."""

    @staticmethod
    def create_dataset(
        session: Session,
        name: str,
        slug: str,
        version: str = "1.0.0",
        description: str | None = None,
        supported_languages: list[str] | None = None,
        dataset_type: str = "VALIDATED_BENCHMARK",
        source_uri: str | None = None,
    ) -> Dataset:
        """Create a new versioned evaluation dataset."""
        langs = supported_languages or ["en", "bn", "bn-Latn", "hi", "ar"]
        hash_payload = f"{name}:{slug}:{version}:{','.join(sorted(langs))}"
        content_hash = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()

        dataset = Dataset(
            name=name,
            slug=slug,
            version=version,
            description=description,
            dataset_type=dataset_type,
            supported_languages=langs,
            source_uri=source_uri,
            content_hash_sha256=content_hash,
            is_locked=False,
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
        return dataset

    @staticmethod
    def add_query_group(
        session: Session,
        dataset_id: uuid.UUID,
        canonical_intent: str,
        domain: str = "general",
    ) -> QueryGroup:
        """Add a semantic query group representing an invariant information need."""
        group = QueryGroup(
            dataset_id=dataset_id,
            canonical_intent=canonical_intent,
            domain=domain,
        )
        session.add(group)
        session.commit()
        session.refresh(group)
        return group

    @staticmethod
    def add_query(
        session: Session,
        query_group_id: uuid.UUID,
        dataset_id: uuid.UUID,
        query_text: str,
        language_code: str,
        script_code: str,
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Query:
        """Add a language/script-specific query realization."""
        query = Query(
            query_group_id=query_group_id,
            dataset_id=dataset_id,
            query_text=query_text,
            language_code=language_code,
            script_code=script_code,
            token_count=token_count,
            metadata_json=metadata,
        )
        session.add(query)
        session.commit()
        session.refresh(query)
        return query

    @staticmethod
    def add_document(
        session: Session,
        dataset_id: uuid.UUID,
        doc_external_id: str,
        content: str,
        language_code: str,
        script_code: str,
        title: str | None = None,
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Add a text document/passage to a dataset."""
        doc = Document(
            dataset_id=dataset_id,
            doc_external_id=doc_external_id,
            title=title,
            content=content,
            language_code=language_code,
            script_code=script_code,
            token_count=token_count,
            metadata_json=metadata,
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
        return doc

    @staticmethod
    def add_relevance_judgment(
        session: Session,
        dataset_id: uuid.UUID,
        query_id: uuid.UUID,
        document_id: uuid.UUID,
        relevance_score: int,
        origin: str = "HUMAN_EXPERT",
    ) -> RelevanceJudgment:
        """Add a ground-truth graded relevance rating (0 to 3)."""
        if not (0 <= relevance_score <= 3):
            raise ValueError(f"Relevance score must be between 0 and 3, got {relevance_score}")

        judgment = RelevanceJudgment(
            dataset_id=dataset_id,
            query_id=query_id,
            document_id=document_id,
            relevance_score=relevance_score,
            judgment_origin=origin,
        )
        session.add(judgment)
        session.commit()
        session.refresh(judgment)
        return judgment

    @staticmethod
    def store_document_embedding(
        session: Session,
        document_id: uuid.UUID,
        model_id: str,
        embedding: np.ndarray,
        is_normalized: bool = True,
    ) -> DocumentEmbedding:
        """Store an embedding vector for a document."""
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        dim = vec.shape[0]

        doc_embedding = DocumentEmbedding(
            document_id=document_id,
            model_id=model_id,
            embedding=vec,
            dimension=dim,
            is_normalized=is_normalized,
        )
        session.add(doc_embedding)
        session.commit()
        session.refresh(doc_embedding)
        return doc_embedding

    @staticmethod
    def index_dataset_with_model(
        session: Session,
        dataset_id: uuid.UUID,
        model_adapter: BaseEmbeddingAdapter,
        batch_size: int = 32,
    ) -> int:
        """Compute and persist embeddings for all documents in a dataset using an adapter.

        Returns the number of documents indexed.
        """
        stmt = (
            select(Document)
            .where(Document.dataset_id == dataset_id)
            .order_by(Document.created_at.asc())
        )
        docs = session.execute(stmt).scalars().all()

        if not docs:
            return 0

        texts = [d.content for d in docs]
        vectors = model_adapter.encode_documents(texts, batch_size=batch_size, normalize=True)

        for doc, vec in zip(docs, vectors):
            doc_emb = DocumentEmbedding(
                document_id=doc.id,
                model_id=model_adapter.model_id,
                embedding=vec,
                dimension=vec.shape[0],
                is_normalized=True,
            )
            session.add(doc_emb)

        session.commit()
        return len(docs)
