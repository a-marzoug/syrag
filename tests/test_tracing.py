from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import cast

import pytest
from httpx import ASGITransport, AsyncClient

from syrag.app import create_app
from syrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore, PassThroughChunker
from syrag.schemas import DocumentChunk, IngestRequest, QueryRequest, SourceDocument
from syrag.tracing import OpenTelemetrySpan, OpenTelemetryTracer, OpenTelemetryTracing


@dataclass(slots=True)
class RecordedSpan:
    name: str
    parent_name: str | None
    attributes: dict[str, object] = field(default_factory=dict)
    status_code: str | None = None
    ended: bool = False
    exceptions: list[str] = field(default_factory=list)

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status: object) -> None:
        self.status_code = getattr(getattr(status, "status_code", None), "name", None)

    def record_exception(self, exception: BaseException) -> None:
        self.exceptions.append(type(exception).__name__)

    def end(self) -> None:
        self.ended = True


class FakeTracer(OpenTelemetryTracer):
    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []
        self._current_stack: list[RecordedSpan] = []

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, object] | None = None,
    ) -> OpenTelemetrySpan:
        span = RecordedSpan(
            name=name,
            parent_name=self._current_stack[-1].name if self._current_stack else None,
        )
        if attributes is not None:
            span.attributes.update(dict(attributes))
        self.spans.append(span)
        return span

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, object] | None = None,
    ) -> Iterator[OpenTelemetrySpan]:
        span = cast(RecordedSpan, self.start_span(name, attributes=attributes))
        self._current_stack.append(span)
        try:
            yield span
        finally:
            self._current_stack.pop()
            span.end()


@pytest.mark.asyncio
async def test_query_route_emits_request_and_stage_spans() -> None:
    app = create_app()
    tracer = FakeTracer()
    app.configure_tracing(tracing=OpenTelemetryTracing(tracer=tracer))
    embedder = InMemoryEmbedder()
    vector_store = InMemoryVectorStore()
    llm = InMemoryLLM()

    embeddings = await embedder.embed(
        [
            "SyRAG is a production-first Python framework for RAG services.",
            "It emphasizes observability, type safety, and multi-tenancy.",
        ]
    )
    await vector_store.upsert(
        chunks=[
            DocumentChunk(
                chunk_id="overview-1-chunk-0",
                source_id="overview-1",
                content="SyRAG is a production-first Python framework for RAG services.",
                metadata={"source_id": "overview-1", "page_number": 1},
                page_number=1,
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="overview-2-chunk-0",
                source_id="overview-2",
                content="It emphasizes observability, type safety, and multi-tenancy.",
                metadata={"source_id": "overview-2", "page_number": 1},
                page_number=1,
                chunk_index=0,
            ),
        ],
        embeddings=embeddings,
        collection="overview",
    )

    @app.query(
        "/query",
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
    )
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/query",
            json={"query": "What is SyRAG?", "collection": "overview"},
            headers={"x-request-id": "request-123", "x-tenant-id": "tenant-a"},
        )

    assert response.status_code == 200
    assert [span.name for span in tracer.spans] == [
        "syrag.request",
        "syrag.query.embed",
        "syrag.query.retrieve",
        "syrag.query.assemble",
        "syrag.query.policy",
        "syrag.query.generate",
    ]
    request_span = tracer.spans[0]
    assert request_span.attributes["syrag.request_id"] == "request-123"
    assert request_span.attributes["syrag.tenant_id"] == "tenant-a"
    assert request_span.attributes["http.response.status_code"] == 200
    assert request_span.ended is True

    for span in tracer.spans[1:]:
        assert span.parent_name == "syrag.request"
        assert span.ended is True
        assert span.attributes["syrag.span.kind"] == "stage"


@pytest.mark.asyncio
async def test_ingest_route_failure_marks_request_and_stage_spans_as_error() -> None:
    class FailingChunker(PassThroughChunker):
        async def chunk(
            self,
            documents: Sequence[SourceDocument],
        ) -> list[DocumentChunk]:
            raise RuntimeError("chunker crashed")

    app = create_app()
    tracer = FakeTracer()
    app.configure_tracing(tracing=OpenTelemetryTracing(tracer=tracer))

    @app.ingest(
        "/ingest",
        chunker=FailingChunker(),
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
    )
    async def ingest_route(request: IngestRequest) -> IngestRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/ingest", json={"documents": ["SyRAG doc"]})

    assert response.status_code == 500
    request_span = tracer.spans[0]
    stage_span = tracer.spans[1]
    assert request_span.status_code == "ERROR"
    assert stage_span.status_code == "ERROR"
    assert request_span.attributes["http.response.status_code"] == 500
    assert stage_span.attributes["syrag.stage"] == "chunk"
