from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from math import ceil
from time import monotonic

from starlette.requests import Request

from fastrag.errors import RateLimitExceededError, SafetyGuardError
from fastrag.schemas import IngestRequest, QueryRequest, RequestContext


class InMemoryRateLimiter:
    """Simple in-process sliding-window rate limiter for HTTP requests."""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float = 60.0,
        include_path: bool = True,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_requests < 1:
            msg = "max_requests must be at least 1"
            raise ValueError(msg)
        if window_seconds <= 0:
            msg = "window_seconds must be greater than 0"
            raise ValueError(msg)

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.include_path = include_path
        self._clock = clock
        self._lock = asyncio.Lock()
        self._buckets: dict[str, deque[float]] = {}

    async def check(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> None:
        key = self._resolve_key(request=request, context=context)
        now = self._clock()
        window_start = now - self.window_seconds

        async with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            while bucket and bucket[0] <= window_start:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                retry_after_seconds = max(1, ceil(bucket[0] + self.window_seconds - now))
                raise RateLimitExceededError(
                    code="rate_limited",
                    message="Request rate limit exceeded.",
                    details={
                        "max_requests": self.max_requests,
                        "window_seconds": self.window_seconds,
                        "retry_after_seconds": retry_after_seconds,
                    },
                )

            bucket.append(now)

    def _resolve_key(self, *, request: Request, context: RequestContext) -> str:
        identity = (
            context.subject_id
            or context.tenant_id
            or self._resolve_client_ip(request)
            or "anonymous"
        )
        if not self.include_path:
            return identity
        return f"{request.method}:{request.url.path}:{identity}"

    def _resolve_client_ip(self, request: Request) -> str | None:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",", maxsplit=1)[0].strip() or None
        if request.client is None:
            return None
        return request.client.host


class DefaultSafetyGuard:
    """Rejects oversized or suspicious request payloads before pipeline work begins."""

    def __init__(
        self,
        *,
        max_query_characters: int = 4_000,
        max_query_filters: int = 20,
        max_ingest_documents: int = 100,
        max_document_characters: int = 20_000,
        max_total_ingest_characters: int = 200_000,
        max_ingest_metadata_entries: int = 50,
    ) -> None:
        self.max_query_characters = max_query_characters
        self.max_query_filters = max_query_filters
        self.max_ingest_documents = max_ingest_documents
        self.max_document_characters = max_document_characters
        self.max_total_ingest_characters = max_total_ingest_characters
        self.max_ingest_metadata_entries = max_ingest_metadata_entries

    async def validate_query(
        self,
        *,
        request: Request,
        payload: QueryRequest,
        context: RequestContext,
    ) -> QueryRequest:
        del request, context

        query_length = len(payload.query)
        if query_length > self.max_query_characters:
            raise SafetyGuardError(
                code="query_too_large",
                message="Query exceeds the configured safety limit.",
                details={
                    "max_query_characters": self.max_query_characters,
                    "actual_query_characters": query_length,
                },
            )

        filter_count = len(payload.filters)
        if filter_count > self.max_query_filters:
            raise SafetyGuardError(
                code="too_many_filters",
                message="Query filters exceed the configured safety limit.",
                details={
                    "max_query_filters": self.max_query_filters,
                    "actual_query_filters": filter_count,
                },
            )

        return payload

    async def validate_ingest(
        self,
        *,
        request: Request,
        payload: IngestRequest,
        context: RequestContext,
    ) -> IngestRequest:
        del request, context

        document_count = len(payload.documents)
        if document_count > self.max_ingest_documents:
            raise SafetyGuardError(
                code="too_many_documents",
                message="Ingest request exceeds the configured document safety limit.",
                details={
                    "max_ingest_documents": self.max_ingest_documents,
                    "actual_ingest_documents": document_count,
                },
            )

        metadata_entries = len(payload.metadata)
        if metadata_entries > self.max_ingest_metadata_entries:
            raise SafetyGuardError(
                code="metadata_too_large",
                message="Ingest metadata exceeds the configured safety limit.",
                details={
                    "max_ingest_metadata_entries": self.max_ingest_metadata_entries,
                    "actual_ingest_metadata_entries": metadata_entries,
                },
            )

        total_characters = 0
        for index, document in enumerate(payload.documents):
            document_length = len(document)
            if document_length > self.max_document_characters:
                raise SafetyGuardError(
                    code="document_too_large",
                    message="An ingest document exceeds the configured safety limit.",
                    details={
                        "document_index": index,
                        "max_document_characters": self.max_document_characters,
                        "actual_document_characters": document_length,
                    },
                )
            total_characters += document_length

        if total_characters > self.max_total_ingest_characters:
            raise SafetyGuardError(
                code="ingest_too_large",
                message="Ingest payload exceeds the configured total safety limit.",
                details={
                    "max_total_ingest_characters": self.max_total_ingest_characters,
                    "actual_total_ingest_characters": total_characters,
                },
            )

        return payload
