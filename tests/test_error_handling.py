from collections.abc import Sequence

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from fastrag.app import create_app
from fastrag.errors import FastRAGError
from fastrag.protocols import (
    LLM,
    AuthHook,
    Embedder,
    EmbeddingVector,
    Filters,
    GenerationPolicy,
    VectorStore,
)
from fastrag.schemas import (
    AssembledPrompt,
    DocumentChunk,
    GenerationRequest,
    IngestRequest,
    QueryRequest,
    RAGResponse,
    RequestContext,
    RetrievedChunk,
)


class FailingEmbedder(Embedder):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("embedder crashed")


class PassthroughEmbedder(Embedder):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class PassthroughVectorStore(VectorStore):
    async def upsert(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
        collection: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        return None

    async def query(
        self,
        *,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: Filters | None = None,
    ) -> list[RetrievedChunk]:
        return []


class PassthroughLLM(LLM):
    async def generate(
        self,
        *,
        generation: GenerationRequest,
    ) -> RAGResponse:
        return RAGResponse(answer=generation.query.query)


class FailingGenerationPolicy(GenerationPolicy):
    async def apply(
        self,
        *,
        prompt: AssembledPrompt,
    ) -> GenerationRequest:
        raise RuntimeError("generation policy crashed")


class FailingAuthHook(AuthHook):
    async def authenticate(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> RequestContext:
        raise FastRAGError(
            code="authentication_failed",
            message="Failed to authenticate the request.",
            stage="auth",
            status_code=401,
            details={"component": type(self).__name__},
        )


@pytest.mark.asyncio
async def test_query_failures_return_stage_aware_error_response() -> None:
    app = create_app()

    @app.query(
        "/query",
        embedder=FailingEmbedder(),
        vector_store=PassthroughVectorStore(),
        llm=PassthroughLLM(),
    )
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/query", json={"query": "What is FastRAG?"})

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "embedding_failed",
            "message": "Failed to embed the query.",
            "stage": "embed",
            "details": {"component": "FailingEmbedder"},
        }
    }


@pytest.mark.asyncio
async def test_ingest_failures_return_stage_aware_error_response() -> None:
    app = create_app()

    @app.ingest(
        "/ingest",
        embedder=FailingEmbedder(),
        vector_store=PassthroughVectorStore(),
    )
    async def ingest_route(request: IngestRequest) -> IngestRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/ingest", json={"documents": ["FastRAG doc"]})

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "embedding_failed",
            "message": "Failed to embed the ingest documents.",
            "stage": "embed",
            "details": {"component": "FailingEmbedder"},
        }
    }


@pytest.mark.asyncio
async def test_generation_policy_failures_return_stage_aware_error_response() -> None:
    app = create_app()

    @app.query(
        "/query",
        embedder=PassthroughEmbedder(),
        vector_store=PassthroughVectorStore(),
        llm=PassthroughLLM(),
        generation_policy=FailingGenerationPolicy(),
    )
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/query", json={"query": "What is FastRAG?"})

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "generation_policy_failed",
            "message": "Failed to apply the generation policy.",
            "stage": "policy",
            "details": {"component": "FailingGenerationPolicy"},
        }
    }


@pytest.mark.asyncio
async def test_auth_hook_failures_return_stage_aware_error_response() -> None:
    app = create_app()
    app.set_auth_hook(FailingAuthHook())

    @app.query(
        "/query",
        embedder=PassthroughEmbedder(),
        vector_store=PassthroughVectorStore(),
        llm=PassthroughLLM(),
    )
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/query", json={"query": "What is FastRAG?"})

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "code": "authentication_failed",
            "message": "Failed to authenticate the request.",
            "stage": "auth",
            "details": {"component": "FailingAuthHook"},
        }
    }


@pytest.mark.asyncio
async def test_tenant_mismatch_returns_structured_error_response() -> None:
    app = create_app()

    @app.query(
        "/query",
        embedder=PassthroughEmbedder(),
        vector_store=PassthroughVectorStore(),
        llm=PassthroughLLM(),
    )
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/query",
            json={"query": "What is FastRAG?", "tenant_id": "tenant-b"},
            headers={"x-tenant-id": "tenant-a"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "tenant_mismatch",
            "message": "Request tenant does not match the scoped tenant context.",
            "stage": "request",
            "details": {
                "context_tenant_id": "tenant-a",
                "request_tenant_id": "tenant-b",
            },
        }
    }
