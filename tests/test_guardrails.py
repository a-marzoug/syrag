from collections.abc import Sequence

import pytest
from httpx import ASGITransport, AsyncClient

from fastrag.app import create_app
from fastrag.guardrails import DefaultSafetyGuard, InMemoryRateLimiter
from fastrag.protocols import Embedder
from fastrag.providers import InMemoryLLM, InMemoryVectorStore
from fastrag.schemas import IngestRequest, QueryRequest


class StaticEmbedder(Embedder):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


@pytest.mark.asyncio
async def test_rate_limiter_returns_structured_429_response() -> None:
    app = create_app()
    current_time = [0.0]
    app.set_rate_limiter(
        InMemoryRateLimiter(
            max_requests=1,
            window_seconds=30.0,
            clock=lambda: current_time[0],
        )
    )

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        first_response = await client.get("/health")
        second_response = await client.get("/health")

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert "x-request-id" in second_response.headers
    assert second_response.headers["retry-after"] == "30"
    assert second_response.json() == {
        "error": {
            "code": "rate_limited",
            "message": "Request rate limit exceeded.",
            "stage": "rate_limit",
            "details": {
                "max_requests": 1,
                "window_seconds": 30.0,
                "retry_after_seconds": 30,
            },
        }
    }


@pytest.mark.asyncio
async def test_query_safety_guard_rejects_oversized_queries() -> None:
    app = create_app()
    app.set_safety_guard(DefaultSafetyGuard(max_query_characters=5))

    @app.query(
        "/query",
        embedder=StaticEmbedder(),
        vector_store=InMemoryVectorStore(),
        llm=InMemoryLLM(),
    )
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/query", json={"query": "too long"})

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "query_too_large",
            "message": "Query exceeds the configured safety limit.",
            "stage": "safety",
            "details": {
                "max_query_characters": 5,
                "actual_query_characters": 8,
            },
        }
    }


@pytest.mark.asyncio
async def test_ingest_safety_guard_rejects_excessive_document_counts() -> None:
    app = create_app()
    app.set_safety_guard(DefaultSafetyGuard(max_ingest_documents=1))

    @app.ingest(
        "/ingest",
        embedder=StaticEmbedder(),
        vector_store=InMemoryVectorStore(),
    )
    async def ingest_route(request: IngestRequest) -> IngestRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/ingest", json={"documents": ["doc-1", "doc-2"]})

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "too_many_documents",
            "message": "Ingest request exceeds the configured document safety limit.",
            "stage": "safety",
            "details": {
                "max_ingest_documents": 1,
                "actual_ingest_documents": 2,
            },
        }
    }


def test_guardrail_hooks_must_implement_protocols() -> None:
    app = create_app()

    with pytest.raises(TypeError, match="rate_limiter must implement the RateLimiter protocol"):
        app.set_rate_limiter(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="safety_guard must implement the SafetyGuard protocol"):
        app.set_safety_guard(object())  # type: ignore[arg-type]
