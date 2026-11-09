"""Lightweight in-memory event stream broadcaster for Server-Sent Events (SSE)."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any


class EventStreamManager:
    """Manages asynchronous pub/sub event queues for SSE endpoints without external broker dependencies."""

    def __init__(self) -> None:
        self._subscribers: dict[uuid.UUID, list[asyncio.Queue[str]]] = defaultdict(list)

    def subscribe(self, experiment_id: uuid.UUID) -> asyncio.Queue[str]:
        """Subscribe a new client queue to an experiment's progress stream."""
        queue: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers[experiment_id].append(queue)
        return queue

    def unsubscribe(self, experiment_id: uuid.UUID, queue: asyncio.Queue[str]) -> None:
        """Remove a subscriber queue when client disconnects."""
        if experiment_id in self._subscribers:
            if queue in self._subscribers[experiment_id]:
                self._subscribers[experiment_id].remove(queue)
            if not self._subscribers[experiment_id]:
                del self._subscribers[experiment_id]

    async def broadcast(
        self,
        experiment_id: uuid.UUID,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Broadcast an event to all active subscribers of an experiment."""
        payload = {
            "experiment_id": str(experiment_id),
            "event": event_type,
            **data,
        }
        sse_message = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

        queues = self._subscribers.get(experiment_id, [])
        for queue in list(queues):
            try:
                queue.put_nowait(sse_message)
            except Exception:
                pass


stream_manager = EventStreamManager()
