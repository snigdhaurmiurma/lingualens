"""Lightweight in-memory event stream broadcaster for Server-Sent Events (SSE).

Supports subscription registration, replay buffering for late-joining subscribers,
and automatic connection cleanup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class EventStreamManager:
    """Manages asynchronous pub/sub event queues for SSE endpoints without external broker dependencies.

    Features:
    1. Dynamic subscription and unsubscription per experiment_id.
    2. Replay buffer so late-joining clients (connecting after run initiation) receive early events.
    3. Terminal event detection for graceful stream closure.
    """

    def __init__(self, max_history_per_experiment: int = 100) -> None:
        self.max_history_per_experiment = max_history_per_experiment
        self._subscribers: dict[uuid.UUID, list[asyncio.Queue[str]]] = defaultdict(list)
        self._history: dict[uuid.UUID, list[str]] = defaultdict(list)

    def subscribe(self, experiment_id: uuid.UUID, replay_history: bool = True) -> asyncio.Queue[str]:
        """Subscribe a new client queue to an experiment's progress stream.

        If replay_history is True, previously emitted events for this experiment
        are queued immediately so late-connecting clients do not miss early stages.
        """
        queue: asyncio.Queue[str] = asyncio.Queue()

        if replay_history and experiment_id in self._history:
            for past_event in self._history[experiment_id]:
                queue.put_nowait(past_event)

        self._subscribers[experiment_id].append(queue)
        logger.debug("Subscribed client to experiment '%s' (active: %d)", experiment_id, len(self._subscribers[experiment_id]))
        return queue

    def unsubscribe(self, experiment_id: uuid.UUID, queue: asyncio.Queue[str]) -> None:
        """Remove a subscriber queue when client disconnects."""
        if experiment_id in self._subscribers:
            if queue in self._subscribers[experiment_id]:
                self._subscribers[experiment_id].remove(queue)
            if not self._subscribers[experiment_id]:
                del self._subscribers[experiment_id]
        logger.debug("Unsubscribed client from experiment '%s'", experiment_id)

    async def broadcast(
        self,
        experiment_id: uuid.UUID,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Broadcast an event to all active subscribers and append to replay history."""
        payload = {
            "experiment_id": str(experiment_id),
            "event": event_type,
            **data,
        }
        sse_message = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

        # Record in replay buffer
        hist = self._history[experiment_id]
        hist.append(sse_message)
        if len(hist) > self.max_history_per_experiment:
            hist.pop(0)

        # Distribute to active subscriber queues
        queues = self._subscribers.get(experiment_id, [])
        for queue in list(queues):
            try:
                queue.put_nowait(sse_message)
            except Exception:
                pass

    def clear_history(self, experiment_id: uuid.UUID) -> None:
        """Clear event replay history for an experiment to conserve memory."""
        if experiment_id in self._history:
            del self._history[experiment_id]


stream_manager = EventStreamManager()
