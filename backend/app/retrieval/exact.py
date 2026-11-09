"""Exact / Brute-force Cosine Similarity Retrieval Engine using vectorized NumPy."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.dataset import Document
from app.models.embedding import DocumentEmbedding
from app.retrieval.base import BaseVectorRetriever, SearchResult


class NumPyExactVectorRetriever(BaseVectorRetriever):
    """Ground-truth exact nearest-neighbor vector retrieval engine.

    Performs exhaustive matrix multiplication across all indexed document vectors.
    Guarantees 100% nearest-neighbor search fidelity (IndexRecall@K = 1.0 against
    the embedding space) with zero index approximation or quantization loss.

    Note on Terminology:
        Exact nearest-neighbor retrieval guarantees finding the geometrically closest
        vectors in the embedding space. It does NOT guarantee 100% Information Retrieval
        Recall (IR Recall@K) against human relevance judgments, which depends on whether
        the embedding model accurately mapped semantically relevant documents near the query.
    """

    def __init__(
        self,
        document_ids: list[uuid.UUID],
        doc_external_ids: list[str],
        embeddings: np.ndarray,
        contents: list[str],
        titles: list[str | None],
        languages: list[str],
        scripts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        dataset_ids: list[uuid.UUID] | None = None,
    ) -> None:
        """Initialize the exact in-memory vector index.

        Args:
            document_ids: List of document UUIDs.
            doc_external_ids: List of human/source document IDs.
            embeddings: 2D numpy array of shape (N, dimension).
            contents: List of passage texts.
            titles: List of passage titles.
            languages: List of language codes.
            scripts: List of script codes.
            metadatas: Optional list of document metadata dictionaries.
            dataset_ids: Optional list of dataset UUIDs.
        """
        if len(document_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(document_ids)} document IDs vs {embeddings.shape[0]} embeddings."
            )

        self.document_ids = list(document_ids)
        self.doc_external_ids = list(doc_external_ids)
        self.contents = list(contents)
        self.titles = list(titles)
        self.languages = list(languages)
        self.scripts = list(scripts)
        self.metadatas = list(metadatas) if metadatas is not None else [{} for _ in document_ids]
        self.dataset_ids = list(dataset_ids) if dataset_ids is not None else [None] * len(document_ids)

        # Ensure float32 and L2 unit normalization for exact cosine inner-product
        if embeddings.size > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            self.matrix = (embeddings / norms).astype(np.float32)
            self.dimension = embeddings.shape[1]
        else:
            self.matrix = np.empty((0, 384), dtype=np.float32)
            self.dimension = 384

    @classmethod
    def from_database(
        cls,
        session: Session,
        model_id: str,
        dataset_id: uuid.UUID | None = None,
    ) -> NumPyExactVectorRetriever:
        """Hydrate an exact vector retriever from persistent database records."""
        query = (
            select(
                Document.id,
                Document.doc_external_id,
                Document.title,
                Document.content,
                Document.language_code,
                Document.script_code,
                Document.metadata_json,
                Document.dataset_id,
                DocumentEmbedding.embedding,
            )
            .join(DocumentEmbedding, Document.id == DocumentEmbedding.document_id)
            .where(DocumentEmbedding.model_id == model_id)
        )

        if dataset_id is not None:
            query = query.where(Document.dataset_id == dataset_id)

        rows = session.execute(query).all()

        if not rows:
            return cls(
                document_ids=[],
                doc_external_ids=[],
                embeddings=np.empty((0, 384), dtype=np.float32),
                contents=[],
                titles=[],
                languages=[],
                scripts=[],
            )

        doc_ids: list[uuid.UUID] = []
        doc_ext_ids: list[str] = []
        titles: list[str | None] = []
        contents: list[str] = []
        langs: list[str] = []
        scripts: list[str] = []
        metas: list[dict[str, Any]] = []
        ds_ids: list[uuid.UUID] = []
        vectors: list[np.ndarray] = []

        for row in rows:
            doc_ids.append(row[0])
            doc_ext_ids.append(row[1])
            titles.append(row[2])
            contents.append(row[3])
            langs.append(row[4])
            scripts.append(row[5])
            metas.append(row[6] or {})
            ds_ids.append(row[7])

            emb_val = row[8]
            if isinstance(emb_val, np.ndarray):
                vectors.append(emb_val)
            elif isinstance(emb_val, list):
                vectors.append(np.array(emb_val, dtype=np.float32))
            else:
                vectors.append(np.array(emb_val, dtype=np.float32))

        embeddings_matrix = np.vstack(vectors)

        return cls(
            document_ids=doc_ids,
            doc_external_ids=doc_ext_ids,
            embeddings=embeddings_matrix,
            contents=contents,
            titles=titles,
            languages=langs,
            scripts=scripts,
            metadatas=metas,
            dataset_ids=ds_ids,
        )

    def count(self) -> int:
        """Return the total number of indexed vectors."""
        return len(self.document_ids)

    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_language: str | None = None,
        filter_dataset_id: uuid.UUID | None = None,
    ) -> list[SearchResult]:
        """Perform exact brute-force cosine similarity retrieval."""
        if self.count() == 0 or top_k <= 0:
            return []

        # Validate input query vector
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {q.shape[0]} does not match index dimension {self.dimension}"
            )

        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            raise ValueError("Cannot search with a zero-vector query.")
        q = q / q_norm

        # Filter candidate indices if language or dataset filter requested
        candidate_indices = list(range(len(self.document_ids)))
        if filter_language is not None:
            candidate_indices = [
                idx for idx in candidate_indices if self.languages[idx] == filter_language
            ]
        if filter_dataset_id is not None:
            candidate_indices = [
                idx for idx in candidate_indices if self.dataset_ids[idx] == filter_dataset_id
            ]

        if not candidate_indices:
            return []

        candidate_matrix = self.matrix[candidate_indices]
        # Exact Cosine Similarity via Dot Product of Unit Vectors
        scores = candidate_matrix @ q

        # Rank candidates descending by score
        k = min(top_k, len(candidate_indices))
        top_order = np.argsort(-scores)[:k]

        results: list[SearchResult] = []
        for rank_idx, local_idx in enumerate(top_order, start=1):
            global_idx = candidate_indices[local_idx]
            sim_score = float(np.clip(scores[local_idx], -1.0, 1.0))

            results.append(
                SearchResult(
                    document_id=self.document_ids[global_idx],
                    doc_external_id=self.doc_external_ids[global_idx],
                    score=sim_score,
                    rank=rank_idx,
                    language_code=self.languages[global_idx],
                    script_code=self.scripts[global_idx],
                    title=self.titles[global_idx],
                    content=self.contents[global_idx],
                    metadata=self.metadatas[global_idx],
                )
            )

        return results
