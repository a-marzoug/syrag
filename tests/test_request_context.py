from collections.abc import Sequence

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from fastrag.app import create_app
from fastrag.protocols import AuthHook, Embedder, RequestContextHook
from fastrag.providers import InMemoryLLM, InMemoryVectorStore
from fastrag.schemas import DocumentChunk, QueryRequest, RequestContext


class FixedRequestContextHook(RequestContextHook):
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.calls: list[str] = []

    async def enrich(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> RequestContext:
        self.calls.append(request.url.path)
        return context.model_copy(
            update={
                "request_id": self.request_id,
                "metadata": {
                    **context.metadata,
                    "method": request.method,
                    "path": request.url.path,
                },
            }
        )


class RecordingAuthHook(AuthHook):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def authenticate(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> RequestContext:
        self.calls.append((request.url.path, context.request_id))
        return context.model_copy(
            update={
                "subject_id": request.headers.get("x-user-id"),
                "auth_scheme": "test",
            }
        )


class StaticEmbedder(Embedder):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


@pytest.mark.asyncio
async def test_request_context_middleware_populates_request_state_and_response_header() -> None:
    app = create_app()

    @app.get("/context")
    async def context_route(request: Request) -> dict[str, object]:
        return app.get_request_context(request).model_dump()

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/context",
            headers={"x-request-id": "request-123", "x-tenant-id": "tenant-123"},
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-123"
    assert response.json() == {
        "request_id": "request-123",
        "tenant_id": "tenant-123",
        "subject_id": None,
        "auth_scheme": None,
        "scopes": [],
        "metadata": {"method": "GET", "path": "/context"},
    }


@pytest.mark.asyncio
async def test_query_routes_run_request_context_and_auth_hooks() -> None:
    app = create_app()
    request_context_hook = FixedRequestContextHook(request_id="query-request-1")
    auth_hook = RecordingAuthHook()
    app.set_request_context_hook(request_context_hook)
    app.set_auth_hook(auth_hook)

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
        response = await client.post(
            "/query",
            json={"query": "What is FastRAG?"},
            headers={"x-user-id": "user-123"},
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "query-request-1"
    assert request_context_hook.calls == ["/query"]
    assert auth_hook.calls == [("/query", "query-request-1")]


@pytest.mark.asyncio
async def test_query_routes_apply_tenant_context_to_handlers_and_pipeline() -> None:
    app = create_app()
    embedder = StaticEmbedder()
    vector_store = InMemoryVectorStore()
    llm = InMemoryLLM()
    seen_tenant_ids: list[str | None] = []

    await vector_store.upsert(
        chunks=[
            DocumentChunk(
                chunk_id="tenant-doc-chunk-0",
                source_id="tenant-doc",
                content="Tenant-scoped FastRAG context.",
                metadata={},
                page_number=1,
                chunk_index=0,
            )
        ],
        embeddings=await embedder.embed(["Tenant-scoped FastRAG context."]),
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
        seen_tenant_ids.append(request.tenant_id)
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/query",
            json={"query": "What is tenant scoped?", "collection": "overview"},
            headers={"x-tenant-id": "tenant-a"},
        )

    assert response.status_code == 200
    assert response.json()["citations"][0]["source_id"] == "tenant-doc"
    assert seen_tenant_ids == ["tenant-a"]


def test_request_context_hooks_must_implement_protocols() -> None:
    app = create_app()

    with pytest.raises(
        TypeError,
        match="request_context_hook must implement the RequestContextHook protocol",
    ):
        app.set_request_context_hook(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="auth_hook must implement the AuthHook protocol"):
        app.set_auth_hook(object())  # type: ignore[arg-type]
