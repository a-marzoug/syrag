from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class PipelineEvent:
    operation: str
    stage: str
    status: str
    component: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


type EventListener = Callable[[PipelineEvent], None]


class ObservabilityHub:
    """Lightweight in-process event hub for pipeline lifecycle hooks."""

    def __init__(self) -> None:
        self._listeners: list[EventListener] = []

    def add_listener(self, listener: EventListener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: EventListener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            return

    def emit(self, event: PipelineEvent) -> None:
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                continue
