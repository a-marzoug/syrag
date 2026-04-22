from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol, runtime_checkable

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from starlette.requests import Request
from starlette.responses import Response

from fastrag.errors import FastRAGError
from fastrag.observability import PipelineEvent
from fastrag.schemas import RequestContext

type AttributeValue = (
    str
    | bool
    | int
    | float
    | Sequence[str]
    | Sequence[bool]
    | Sequence[int]
    | Sequence[float]
)


@runtime_checkable
class OpenTelemetrySpan(Protocol):
    def set_attribute(self, key: str, value: AttributeValue) -> None: ...

    def set_status(self, status: Status) -> None: ...

    def record_exception(self, exception: BaseException) -> None: ...

    def end(self) -> None: ...


@runtime_checkable
class OpenTelemetryTracer(Protocol):
    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> OpenTelemetrySpan: ...

    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> AbstractContextManager[OpenTelemetrySpan]: ...


class OpenTelemetryTracing:
    """Bridges FastRAG stage events and request lifecycle into OpenTelemetry spans."""

    def __init__(
        self,
        *,
        tracer: OpenTelemetryTracer | None = None,
        instrumentation_scope: str = "fastrag",
    ) -> None:
        self.tracer = tracer or trace.get_tracer(instrumentation_scope)
        self.listener = _OpenTelemetryStageListener(self.tracer)

    @contextmanager
    def start_request_span(self, *, request: Request) -> Iterator[OpenTelemetrySpan]:
        with self.tracer.start_as_current_span(
            "fastrag.request",
            attributes={
                "fastrag.span.kind": "request",
                "http.request.method": request.method,
                "url.path": request.url.path,
            },
        ) as span:
            yield span

    def enrich_request_span(
        self,
        *,
        span: OpenTelemetrySpan,
        context: RequestContext,
    ) -> None:
        self._set_attribute(span, "fastrag.request_id", context.request_id)
        self._set_attribute(span, "fastrag.tenant_id", context.tenant_id)
        self._set_attribute(span, "fastrag.subject_id", context.subject_id)
        self._set_attribute(span, "fastrag.auth_scheme", context.auth_scheme)
        if context.scopes:
            span.set_attribute("fastrag.scopes", list(context.scopes))

    def finish_request_span(
        self,
        *,
        span: OpenTelemetrySpan,
        response: Response | None,
    ) -> None:
        if response is None:
            span.set_status(Status(StatusCode.ERROR))
            return

        span.set_attribute("http.response.status_code", response.status_code)
        if response.status_code >= 400:
            span.set_status(Status(StatusCode.ERROR))

    def record_exception(
        self,
        *,
        span: OpenTelemetrySpan,
        exception: BaseException,
    ) -> None:
        span.record_exception(exception)
        span.set_status(Status(StatusCode.ERROR))
        if isinstance(exception, FastRAGError):
            span.set_attribute("fastrag.error.code", exception.code)
            span.set_attribute("fastrag.error.stage", exception.stage)

    def _set_attribute(
        self,
        span: OpenTelemetrySpan,
        key: str,
        value: str | None,
    ) -> None:
        if value is not None:
            span.set_attribute(key, value)


class _OpenTelemetryStageListener:
    def __init__(self, tracer: OpenTelemetryTracer) -> None:
        self.tracer = tracer
        self._active_spans: dict[tuple[str, str, str, str | None], OpenTelemetrySpan] = {}

    def __call__(self, event: PipelineEvent) -> None:
        key = (
            self._execution_scope_id(),
            event.operation,
            event.stage,
            event.component,
        )
        if event.status == "started":
            self._active_spans[key] = self.tracer.start_span(
                f"fastrag.{event.operation}.{event.stage}",
                attributes=self._attributes_for(event),
            )
            return

        span = self._active_spans.pop(key, None)
        if span is None:
            return

        for attribute_key, attribute_value in self._attributes_for(event).items():
            span.set_attribute(attribute_key, attribute_value)
        if event.status == "failed":
            span.set_status(Status(StatusCode.ERROR))
        span.end()

    def _execution_scope_id(self) -> str:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is not None:
            return f"task:{id(task)}"
        return f"thread:{threading.get_ident()}"

    def _attributes_for(self, event: PipelineEvent) -> dict[str, AttributeValue]:
        attributes: dict[str, AttributeValue] = {
            "fastrag.span.kind": "stage",
            "fastrag.operation": event.operation,
            "fastrag.stage": event.stage,
            "fastrag.status": event.status,
        }
        if event.component is not None:
            attributes["fastrag.component"] = event.component
        for detail_key, detail_value in event.details.items():
            attribute_value = self._normalize_attribute_value(detail_value)
            if attribute_value is not None:
                attributes[f"fastrag.detail.{detail_key}"] = attribute_value
        return attributes

    def _normalize_attribute_value(self, value: Any) -> AttributeValue | None:
        if isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            normalized_items = [self._normalize_sequence_item(item) for item in value]
            if normalized_items and all(item is not None for item in normalized_items):
                return normalized_items  # type: ignore[return-value]
        if value is None:
            return None
        return str(value)

    def _normalize_sequence_item(self, value: Any) -> str | bool | int | float | None:
        if isinstance(value, (str, bool, int, float)):
            return value
        return None
