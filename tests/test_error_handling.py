from collections.abc import Sequence

import pytest
from httpx import ASGITransport, AsyncClient

from fastrag.app import create_app
from fastrag.protocols import LLM, Embedder, EmbeddingVector, Filters, VectorStore
from fastrag.schemas import DocumentChunk, IngestRequest, QueryRequest, RAGResponse, RetrievedChunk


class FailingEmbedder(Embedder):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("embedder crashed")


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
        query: QueryRequest,
        context: Sequence[RetrievedChunk],
    ) -> RAGResponse:
        return RAGResponse(answer=query.query)


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
