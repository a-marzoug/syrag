from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from syrag.observability import PipelineEvent
from syrag.schemas import RequestContext


class JSONLogFormatter(logging.Formatter):
    """Formats SyRAG log records as JSON for structured log sinks."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        syrag_payload = getattr(record, "syrag", None)
        if isinstance(syrag_payload, Mapping):
            payload.update(dict(syrag_payload))
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


class StructuredLogging:
    """Bridges framework lifecycle events into structured stdlib log records."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("syrag")
        self.listener = _StructuredLoggingListener(self.logger)

    def log_request(
        self,
        *,
        request: Request,
        context: RequestContext,
        response: Response | None,
        duration_ms: float,
    ) -> None:
        status_code = response.status_code if response is not None else None
        level = self._request_level(status_code)
        status = "completed"
        if status_code is None or status_code >= 400:
            status = "failed"
        self.logger.log(
            level,
            "SyRAG request completed" if status == "completed" else "SyRAG request failed",
            extra={
                "syrag": {
                    "event": "syrag.request",
                    "status": status,
                    "method": request.method,
                    "path": request.url.path,
                    "request_id": context.request_id,
                    "tenant_id": context.tenant_id,
                    "subject_id": context.subject_id,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 3),
                }
            },
        )

    def _request_level(self, status_code: int | None) -> int:
        if status_code is None or status_code >= 500:
            return logging.ERROR
        if status_code >= 400:
            return logging.WARNING
        return logging.INFO


class _StructuredLoggingListener:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def __call__(self, event: PipelineEvent) -> None:
        level = logging.ERROR if event.status == "failed" else logging.INFO
        message = "SyRAG pipeline stage failed"
        if event.status != "failed":
            message = "SyRAG pipeline event"
        self.logger.log(
            level,
            message,
            extra={
                "syrag": {
                    "event": "syrag.pipeline",
                    "operation": event.operation,
                    "stage": event.stage,
                    "status": event.status,
                    "component": event.component,
                    "details": dict(event.details),
                }
            },
        )
