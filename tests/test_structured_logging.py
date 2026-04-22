from __future__ import annotations

import io
import json
import logging
from collections.abc import Sequence
from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

from fastrag.app import create_app
from fastrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from fastrag.schemas import DocumentChunk, QueryRequest
from fastrag.structured_logging import JSONLogFormatter, StructuredLogging


def _fastrag_payload(record: logging.LogRecord) -> dict[str, Any]:
    typed_record = cast(Any, record)
    return cast(dict[str, Any], typed_record.fastrag)


@pytest.mark.asyncio
async def test_query_route_emits_structured_request_and_pipeline_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app()
    logger = logging.getLogger("fastrag.test.logging.query")
    app.configure_logging(structured_logging=StructuredLogging(logger=logger))
    embedder = InMemoryEmbedder()
    vector_store = InMemoryVectorStore()
    llm = InMemoryLLM()
    caplog.set_level(logging.INFO, logger=logger.name)

    embeddings = await embedder.embed(
        [
            "FastRAG is a production-first Python framework for RAG services.",
            "It emphasizes observability, type safety, and multi-tenancy.",
        ]
    )
    await vector_store.upsert(
        chunks=[
            DocumentChunk(
                chunk_id="overview-1-chunk-0",
                source_id="overview-1",
                content="FastRAG is a production-first Python framework for RAG services.",
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
        tenant_id="tenant-a",
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
            json={"query": "What is FastRAG?", "collection": "overview"},
            headers={"x-request-id": "request-123", "x-tenant-id": "tenant-a"},
        )

    assert response.status_code == 200
    fastrag_records = [
        record
        for record in caplog.records
        if record.name == logger.name and hasattr(record, "fastrag")
    ]

    assert any(
        _fastrag_payload(record) == {
            "event": "fastrag.pipeline",
            "operation": "query",
            "stage": "embed",
            "status": "started",
            "component": "InMemoryEmbedder",
            "details": {},
        }
        for record in fastrag_records
    )

    request_record = next(
        record
        for record in fastrag_records
        if _fastrag_payload(record).get("event") == "fastrag.request"
    )
    request_payload = _fastrag_payload(request_record)
    assert request_payload["status"] == "completed"
    assert request_payload["request_id"] == "request-123"
    assert request_payload["tenant_id"] == "tenant-a"
    assert request_payload["status_code"] == 200
    assert request_payload["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_failed_query_emits_error_level_structured_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingEmbedder(InMemoryEmbedder):
        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise RuntimeError("embedder crashed")

    app = create_app()
    logger = logging.getLogger("fastrag.test.logging.failure")
    app.configure_logging(structured_logging=StructuredLogging(logger=logger))
    caplog.set_level(logging.INFO, logger=logger.name)

    @app.query(
        "/query",
        embedder=FailingEmbedder(),
        vector_store=InMemoryVectorStore(),
        llm=InMemoryLLM(),
    )
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/query", json={"query": "What is FastRAG?"})

    assert response.status_code == 500
    fastrag_records = [
        record
        for record in caplog.records
        if record.name == logger.name and hasattr(record, "fastrag")
    ]
    failed_stage_record = next(
        record
        for record in fastrag_records
        if _fastrag_payload(record).get("event") == "fastrag.pipeline"
        and _fastrag_payload(record).get("stage") == "embed"
        and _fastrag_payload(record).get("status") == "failed"
    )
    assert failed_stage_record.levelno == logging.ERROR
    assert _fastrag_payload(failed_stage_record)["details"] == {"error_type": "RuntimeError"}

    request_record = next(
        record
        for record in fastrag_records
        if _fastrag_payload(record).get("event") == "fastrag.request"
    )
    assert request_record.levelno == logging.ERROR
    request_payload = _fastrag_payload(request_record)
    assert request_payload["status"] == "failed"
    assert request_payload["status_code"] == 500


def test_json_log_formatter_serializes_structured_payloads() -> None:
    logger = logging.getLogger("fastrag.test.logging.formatter")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONLogFormatter())
    logger.addHandler(handler)

    try:
        logger.info(
            "FastRAG request completed",
            extra={
                "fastrag": {
                    "event": "fastrag.request",
                    "status": "completed",
                    "status_code": 200,
                }
            },
        )
    finally:
        logger.removeHandler(handler)
        logger.propagate = True

    payload = json.loads(stream.getvalue())
    assert payload["event"] == "fastrag.request"
    assert payload["status"] == "completed"
    assert payload["status_code"] == 200
    assert payload["level"] == "INFO"
    assert payload["message"] == "FastRAG request completed"
