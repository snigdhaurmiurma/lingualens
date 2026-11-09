"""Experiment execution, progress streaming (SSE), and evaluation reporting endpoints."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query as QueryParam, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import SessionLocal, get_db
from app.models.dataset import Dataset
from app.models.experiment import Experiment, ExperimentRun
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentEvaluationSummary,
    ExperimentResponse,
    ExperimentRunRequest,
    ExperimentRunResponse,
    PaginatedExperimentResponse,
)
from app.services.event_stream import stream_manager
from app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/experiments", tags=["Experiments & Evaluation"])


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(payload: ExperimentCreate, db: Session = Depends(get_db)) -> ExperimentResponse:
    """Create a new evaluation experiment configuration."""
    dataset = db.get(Dataset, payload.dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target dataset with ID '{payload.dataset_id}' does not exist.",
        )

    exp = ExperimentService.create_experiment(
        session=db,
        name=payload.name,
        dataset_id=payload.dataset_id,
        baseline_language_code=payload.baseline_language_code,
        description=payload.description,
        cutoff_k_values=payload.cutoff_k_values,
        similarity_metric=payload.similarity_metric,
    )

    return ExperimentResponse(
        id=exp.id,
        name=exp.name,
        description=exp.description,
        dataset_id=exp.dataset_id,
        baseline_language_code=exp.baseline_language_code,
        cutoff_k_values=exp.cutoff_k_values,
        similarity_metric=exp.similarity_metric,
        created_at=exp.created_at,
        runs=[],
    )


@router.get("", response_model=PaginatedExperimentResponse, status_code=status.HTTP_200_OK)
def list_experiments(
    limit: int = QueryParam(default=50, ge=1, le=100),
    offset: int = QueryParam(default=0, ge=0),
    db: Session = Depends(get_db),
) -> PaginatedExperimentResponse:
    """List experiments with pagination."""
    total = db.scalar(select(func.count()).select_from(Experiment)) or 0
    experiments = ExperimentService.list_experiments(session=db, limit=limit, offset=offset)

    responses: list[ExperimentResponse] = []
    for exp in experiments:
        run_items = [
            ExperimentRunResponse(
                id=r.id,
                experiment_id=r.experiment_id,
                model_id=r.model_id,
                retrieval_mode=r.retrieval_mode,
                status=r.status,
                progress_percent=r.progress_percent,
                error_message=r.error_message,
                device_used=r.device_used,
                hardware_metadata=r.hardware_metadata,
                total_latency_seconds=r.total_latency_seconds,
                created_at=r.created_at,
            )
            for r in exp.runs
        ]

        responses.append(
            ExperimentResponse(
                id=exp.id,
                name=exp.name,
                description=exp.description,
                dataset_id=exp.dataset_id,
                baseline_language_code=exp.baseline_language_code,
                cutoff_k_values=exp.cutoff_k_values,
                similarity_metric=exp.similarity_metric,
                created_at=exp.created_at,
                runs=run_items,
            )
        )
    return PaginatedExperimentResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=responses,
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse, status_code=status.HTTP_200_OK)
def get_experiment(experiment_id: uuid.UUID, db: Session = Depends(get_db)) -> ExperimentResponse:
    """Retrieve details and execution runs of an experiment."""
    exp = ExperimentService.get_experiment(session=db, experiment_id=experiment_id)
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with ID '{experiment_id}' not found.",
        )

    run_items = [
        ExperimentRunResponse(
            id=r.id,
            experiment_id=r.experiment_id,
            model_id=r.model_id,
            retrieval_mode=r.retrieval_mode,
            status=r.status,
            progress_percent=r.progress_percent,
            error_message=r.error_message,
            device_used=r.device_used,
            hardware_metadata=r.hardware_metadata,
            total_latency_seconds=r.total_latency_seconds,
            created_at=r.created_at,
        )
        for r in exp.runs
    ]

    return ExperimentResponse(
        id=exp.id,
        name=exp.name,
        description=exp.description,
        dataset_id=exp.dataset_id,
        baseline_language_code=exp.baseline_language_code,
        cutoff_k_values=exp.cutoff_k_values,
        similarity_metric=exp.similarity_metric,
        created_at=exp.created_at,
        runs=run_items,
    )


async def _run_in_background(
    engine: Any,
    experiment_id: uuid.UUID,
    model_id: str,
    retrieval_mode: str,
    random_seed: int,
) -> None:
    """Background task executing the run with dedicated session."""
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = session_factory()
    try:
        await ExperimentService.execute_run_async(
            session=db,
            experiment_id=experiment_id,
            model_id=model_id,
            retrieval_mode=retrieval_mode,
            random_seed=random_seed,
        )
    finally:
        db.close()


@router.post("/{experiment_id}/run", response_model=ExperimentRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_experiment_run(
    experiment_id: uuid.UUID,
    payload: ExperimentRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ExperimentRunResponse:
    """Trigger an asynchronous model evaluation run with real-time SSE progress streaming."""
    exp = db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with ID '{experiment_id}' not found.",
        )

    # Initial Run placeholder for immediate HTTP 202 response
    initial_run = ExperimentRun(
        experiment_id=experiment_id,
        model_id=payload.model_id,
        retrieval_mode=payload.retrieval_mode.upper(),
        status="PENDING",
        progress_percent=0.0,
        random_seed=payload.random_seed,
        device_used="cpu",
    )
    db.add(initial_run)
    db.commit()
    db.refresh(initial_run)

    # Dispatch to non-blocking background task
    bind_engine = db.get_bind()
    background_tasks.add_task(
        _run_in_background,
        engine=bind_engine,
        experiment_id=experiment_id,
        model_id=payload.model_id,
        retrieval_mode=payload.retrieval_mode,
        random_seed=payload.random_seed,
    )

    return ExperimentRunResponse(
        id=initial_run.id,
        experiment_id=initial_run.experiment_id,
        model_id=initial_run.model_id,
        retrieval_mode=initial_run.retrieval_mode,
        status="PENDING",
        progress_percent=0.0,
        error_message=None,
        device_used=initial_run.device_used,
        hardware_metadata=initial_run.hardware_metadata,
        total_latency_seconds=None,
        created_at=initial_run.created_at,
    )


@router.get("/{experiment_id}/stream", status_code=status.HTTP_200_OK)
async def stream_experiment_progress(experiment_id: uuid.UUID) -> StreamingResponse:
    """Subscribe to real-time Server-Sent Events (SSE) reporting experiment progress."""
    queue = stream_manager.subscribe(experiment_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Emit initial connection handshake
            yield f"event: CONNECTED\ndata: {{\"experiment_id\": \"{experiment_id}\", \"status\": \"connected\"}}\n\n"
            while True:
                # Wait for next event with keep-alive ping
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield message
                    if message.startswith("event: COMPLETED") or message.startswith("event: FAILED"):
                        break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            stream_manager.unsubscribe(experiment_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{experiment_id}/results", response_model=ExperimentEvaluationSummary, status_code=status.HTTP_200_OK)
def get_experiment_results(
    experiment_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> ExperimentEvaluationSummary:
    """Retrieve full evaluation results, bootstrap confidence intervals, and fairness disparity scores."""
    try:
        return ExperimentService.get_experiment_evaluation_summary(
            session=db,
            experiment_id=experiment_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
