"""Experiment management and evaluation orchestration service."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import numpy as np
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.registry import default_registry
from app.core.bootstrap import bootstrap_confidence_interval, paired_bootstrap_gap_ci, paired_wilcoxon_test
from app.core.fairness import compute_disparity_summary
from app.core.metrics import compute_query_metrics
from app.models.dataset import Dataset, Document, Query, RelevanceJudgment
from app.models.experiment import EvaluationResult, Experiment, ExperimentRun, RetrievedDocument
from app.retrieval.exact import NumPyExactVectorRetriever
from app.retrieval.pgvector import PgVectorRetriever
from app.schemas.experiment import (
    ExperimentEvaluationSummary,
    ExperimentRunResponse,
    FairnessDisparityItem,
    MetricResultItem,
)
from app.services.dataset_service import DatasetService
from app.services.event_stream import stream_manager


class ExperimentService:
    """Orchestrates experiment creation, run execution, and evaluation reporting."""

    @staticmethod
    def create_experiment(
        session: Session,
        name: str,
        dataset_id: uuid.UUID,
        baseline_language_code: str = "en",
        description: str | None = None,
        cutoff_k_values: list[int] | None = None,
        similarity_metric: str = "cosine",
    ) -> Experiment:
        """Create a new evaluation experiment configuration."""
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with ID '{dataset_id}' does not exist.")

        experiment = Experiment(
            name=name,
            description=description,
            dataset_id=dataset_id,
            baseline_language_code=baseline_language_code,
            cutoff_k_values=cutoff_k_values or [1, 5, 10, 20],
            similarity_metric=similarity_metric,
        )
        session.add(experiment)
        session.commit()
        session.refresh(experiment)
        return experiment

    @staticmethod
    def get_experiment(session: Session, experiment_id: uuid.UUID) -> Experiment | None:
        """Fetch an experiment by ID."""
        return session.get(Experiment, experiment_id)

    @staticmethod
    def list_experiments(session: Session, limit: int = 50, offset: int = 0) -> list[Experiment]:
        """List experiments with pagination."""
        stmt = (
            select(Experiment)
            .order_by(Experiment.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(session.execute(stmt).scalars().all())

    @staticmethod
    async def execute_run_async(
        session: Session,
        experiment_id: uuid.UUID,
        model_id: str,
        retrieval_mode: str = "EXACT",
        random_seed: int = 42,
    ) -> ExperimentRun:
        """Asynchronously execute an experiment run with streaming progress updates."""
        experiment = session.get(Experiment, experiment_id)
        if not experiment:
            raise ValueError(f"Experiment with ID '{experiment_id}' does not exist.")

        adapter = default_registry.get_adapter(model_id)

        # Create Run entity
        run = ExperimentRun(
            experiment_id=experiment_id,
            model_id=adapter.model_id,
            retrieval_mode=retrieval_mode.upper(),
            status="RUNNING",
            progress_percent=5.0,
            random_seed=random_seed,
            device_used="cpu",
            hardware_metadata={
                "torch_threads": torch.get_num_threads(),
                "device": "cpu",
            },
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        await stream_manager.broadcast(
            experiment_id=experiment_id,
            event_type="STARTED",
            data={
                "run_id": str(run.id),
                "model_id": adapter.model_id,
                "progress": 5.0,
                "detail": "Experiment execution initialized",
            },
        )

        t_start = time.perf_counter()

        try:
            # 1. Load Dataset
            dataset_id = experiment.dataset_id
            docs = session.execute(
                select(Document).where(Document.dataset_id == dataset_id)
            ).scalars().all()
            queries = session.execute(
                select(Query).where(Query.dataset_id == dataset_id)
            ).scalars().all()
            judgments = session.execute(
                select(RelevanceJudgment).where(RelevanceJudgment.dataset_id == dataset_id)
            ).scalars().all()

            if not docs or not queries:
                raise ValueError("Dataset must contain both documents and queries to run evaluation.")

            await stream_manager.broadcast(
                experiment_id=experiment_id,
                event_type="DATASET_LOADING",
                data={
                    "run_id": str(run.id),
                    "progress": 15.0,
                    "detail": f"Loaded {len(docs)} documents and {len(queries)} queries",
                },
            )

            # 2. Embedding Indexing
            if not adapter.is_loaded:
                adapter.load()

            # Ensure documents are indexed
            DatasetService.index_dataset_with_model(
                session=session,
                dataset_id=dataset_id,
                model_adapter=adapter,
            )

            await stream_manager.broadcast(
                experiment_id=experiment_id,
                event_type="EMBEDDING_GENERATION",
                data={
                    "run_id": str(run.id),
                    "progress": 40.0,
                    "detail": "Corpus vectors indexed in storage layer",
                },
            )

            # 3. Retrieval Execution
            judgments_map: dict[tuple[uuid.UUID, uuid.UUID], int] = {
                (j.query_id, j.document_id): j.relevance_score for j in judgments
            }

            max_k = max(experiment.cutoff_k_values)
            query_metrics_records: list[dict[str, Any]] = []

            if retrieval_mode.upper() == "EXACT":
                retriever = NumPyExactVectorRetriever.from_database(
                    session=session,
                    model_id=adapter.model_id,
                    dataset_id=dataset_id,
                )
            else:
                retriever = PgVectorRetriever(
                    session=session,
                    model_id=adapter.model_id,
                    dimension=adapter.get_metadata().embedding_dimension,
                )

            for idx, q_obj in enumerate(queries, start=1):
                q_vec = adapter.encode_queries([q_obj.query_text], normalize=True)[0]
                search_results = retriever.retrieve(
                    query_vector=q_vec,
                    top_k=max_k,
                    filter_dataset_id=dataset_id,
                )

                # SERP Audit logging
                retrieved_doc_ids: list[str] = []
                for s_res in search_results:
                    rel = judgments_map.get((q_obj.id, s_res.document_id), 0)
                    retrieved_doc_ids.append(str(s_res.document_id))
                    ret_doc = RetrievedDocument(
                        experiment_run_id=run.id,
                        query_id=q_obj.id,
                        document_id=s_res.document_id,
                        rank_position=s_res.rank,
                        similarity_score=s_res.score,
                        assigned_relevance=rel,
                    )
                    session.add(ret_doc)

                # Ground truth map for this query
                q_ground_truth = {
                    str(j.document_id): j.relevance_score
                    for j in judgments
                    if j.query_id == q_obj.id
                }

                q_metrics = compute_query_metrics(
                    retrieved_ids=retrieved_doc_ids,
                    relevant_ratings=q_ground_truth,
                    cutoff_k_values=[1, 5, 10, 20],
                    threshold=2,
                )

                query_metrics_records.append({
                    "query_id": q_obj.id,
                    "query_group_id": q_obj.query_group_id,
                    "language_code": q_obj.language_code,
                    "script_code": q_obj.script_code,
                    **q_metrics,
                })

            await stream_manager.broadcast(
                experiment_id=experiment_id,
                event_type="RETRIEVAL_EXECUTION",
                data={
                    "run_id": str(run.id),
                    "progress": 70.0,
                    "detail": f"Retrieved top-{max_k} candidates across all language variants",
                },
            )

            # 4. Metric & Fairness Calculation
            languages = sorted(list({rec["language_code"] for rec in query_metrics_records}))
            baseline_lang = experiment.baseline_language_code

            # Compute and persist aggregates per language
            for lang in languages:
                lang_recs = [r for r in query_metrics_records if r["language_code"] == lang]
                script = lang_recs[0]["script_code"] if lang_recs else "unknown"

                for m_name in ["ndcg@10", "mrr", "precision@10", "recall@10"]:
                    vals = np.array([r[m_name] for r in lang_recs], dtype=np.float64)
                    mean_val = float(np.mean(vals)) if len(vals) > 0 else 0.0

                    ci_low, ci_high = (None, None)
                    if len(vals) >= 2:
                        _, ci_low, ci_high = bootstrap_confidence_interval(
                            vals,
                            num_bootstrap=200,
                            random_seed=random_seed,
                        )

                    p_val = None
                    is_sig = False
                    if lang != baseline_lang:
                        base_recs = [r for r in query_metrics_records if r["language_code"] == baseline_lang]
                        # Pair by query_group_id
                        paired_base = []
                        paired_target = []
                        for tr in lang_recs:
                            match = next((br for br in base_recs if br["query_group_id"] == tr["query_group_id"]), None)
                            if match:
                                paired_base.append(match[m_name])
                                paired_target.append(tr[m_name])

                        if len(paired_base) >= 5:
                            _, p_val, is_sig = paired_wilcoxon_test(
                                paired_base,
                                paired_target,
                            )

                    eval_res = EvaluationResult(
                        experiment_run_id=run.id,
                        language_code=lang,
                        script_code=script,
                        metric_name=m_name,
                        metric_value=mean_val,
                        ci_lower_95=ci_low,
                        ci_upper_95=ci_high,
                        p_value_vs_baseline=p_val,
                        is_statistically_significant=is_sig,
                    )
                    session.add(eval_res)

            await stream_manager.broadcast(
                experiment_id=experiment_id,
                event_type="METRIC_CALCULATION",
                data={
                    "run_id": str(run.id),
                    "progress": 90.0,
                    "detail": "Computed paired Wilcoxon tests, bootstrap CIs, and disparity metrics",
                },
            )

            # Finalize run
            total_sec = time.perf_counter() - t_start
            run.status = "COMPLETED"
            run.progress_percent = 100.0
            run.total_latency_seconds = round(total_sec, 2)
            session.commit()

            await stream_manager.broadcast(
                experiment_id=experiment_id,
                event_type="COMPLETED",
                data={
                    "run_id": str(run.id),
                    "progress": 100.0,
                    "total_latency_seconds": run.total_latency_seconds,
                    "detail": "Experiment completed successfully",
                },
            )

            return run

        except Exception as exc:
            session.rollback()
            run.status = "FAILED"
            run.error_message = str(exc)
            session.commit()

            await stream_manager.broadcast(
                experiment_id=experiment_id,
                event_type="FAILED",
                data={
                    "run_id": str(run.id),
                    "error": str(exc),
                    "detail": "Experiment run failed",
                },
            )
            raise

    @staticmethod
    def get_experiment_evaluation_summary(
        session: Session,
        experiment_id: uuid.UUID,
    ) -> ExperimentEvaluationSummary:
        """Fetch comprehensive evaluation summary and fairness metrics for an experiment."""
        experiment = session.get(Experiment, experiment_id)
        if not experiment:
            raise ValueError(f"Experiment with ID '{experiment_id}' does not exist.")

        runs = list(session.execute(
            select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id)
        ).scalars().all())

        results = list(session.execute(
            select(EvaluationResult)
            .join(ExperimentRun, EvaluationResult.experiment_run_id == ExperimentRun.id)
            .where(ExperimentRun.experiment_id == experiment_id)
        ).scalars().all())

        metric_items = [
            MetricResultItem(
                language_code=r.language_code,
                script_code=r.script_code,
                metric_name=r.metric_name,
                metric_value=r.metric_value,
                ci_lower_95=r.ci_lower_95,
                ci_upper_95=r.ci_upper_95,
                p_value_vs_baseline=r.p_value_vs_baseline,
                is_statistically_significant=r.is_statistically_significant,
            )
            for r in results
        ]

        # Calculate disparity summary for ndcg@10
        baseline_lang = experiment.baseline_language_code
        disparities: list[FairnessDisparityItem] = []
        ndcg_results = [r for r in results if r.metric_name == "ndcg@10"]
        base_ndcg = next((r.metric_value for r in ndcg_results if r.language_code == baseline_lang), None)

        if base_ndcg is not None:
            for r in ndcg_results:
                if r.language_code == baseline_lang:
                    continue
                gap = round(base_ndcg - r.metric_value, 4)
                ratio = round(r.metric_value / max(1e-9, base_ndcg), 4)

                disparities.append(
                    FairnessDisparityItem(
                        language_code=r.language_code,
                        baseline_language=baseline_lang,
                        absolute_performance_gap=gap,
                        relative_disparity_ratio=ratio,
                        p_value=r.p_value_vs_baseline,
                        is_statistically_significant=r.is_statistically_significant,
                    )
                )

        # Transliteration penalties (bn vs bn-Latn)
        trans_penalties: dict[str, float] = {}
        bn_val = next((r.metric_value for r in ndcg_results if r.language_code == "bn"), None)
        bn_latn_val = next((r.metric_value for r in ndcg_results if r.language_code == "bn-Latn"), None)
        if bn_val is not None and bn_latn_val is not None:
            trans_penalties["bn_to_bn-Latn"] = round(bn_val - bn_latn_val, 4)

        return ExperimentEvaluationSummary(
            experiment_id=experiment.id,
            experiment_name=experiment.name,
            dataset_id=experiment.dataset_id,
            baseline_language=baseline_lang,
            runs=[
                ExperimentRunResponse.model_validate(run, from_attributes=True)
                for run in runs
            ],
            metrics=metric_items,
            fairness_disparities=disparities,
            transliteration_penalties=trans_penalties,
        )
