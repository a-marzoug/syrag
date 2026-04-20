from typing import cast

import pytest
from httpx import ASGITransport, AsyncClient

from fastrag.app import create_app
from fastrag.protocols import Embedder
from fastrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from fastrag.schemas import DocumentChunk, QueryRequest


@pytest.mark.asyncio
async def test_query_decorator_registers_grounded_query_route() -> None:
    app = create_app()
    embedder = InMemoryEmbedder()
    vector_store = InMemoryVectorStore()
    llm = InMemoryLLM()

    documents = [
        "FastRAG is a production-first Python framework for RAG services.",
        "It emphasizes observability, type safety, and multi-tenancy.",
    ]
    embeddings = await embedder.embed(documents)
    await vector_store.upsert(
        chunks=[
            DocumentChunk(
                chunk_id="overview-1-chunk-0",
                source_id="overview-1",
                content=documents[0],
                metadata={"source_id": "overview-1", "page_number": 1},
                page_number=1,
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="overview-2-chunk-0",
                source_id="overview-2",
                content=documents[1],
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
            json={
                "query": "What is FastRAG?",
                "collection": "overview",
                "top_k": 2,
            },
        )

    assert response.status_code == 200
    assert response.json()["citations"][0]["source_id"] == "overview-1"
    assert "FastRAG" in response.json()["answer"]


def test_query_decorator_rejects_invalid_components() -> None:
    app = create_app()

    with pytest.raises(TypeError, match="embedder must implement the Embedder protocol"):
        app.query(
            "/query",
            embedder=cast(Embedder, object()),
            vector_store=InMemoryVectorStore(),
            llm=InMemoryLLM(),
        )
