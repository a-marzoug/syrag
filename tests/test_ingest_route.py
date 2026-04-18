from typing import cast

import pytest
from httpx import ASGITransport, AsyncClient

from fastrag.app import create_app
from fastrag.protocols import Embedder
from fastrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from fastrag.schemas import IngestRequest, QueryRequest


@pytest.mark.asyncio
async def test_ingest_decorator_registers_ingestion_route() -> None:
    app = create_app()
    embedder = InMemoryEmbedder()
    vector_store = InMemoryVectorStore()
    llm = InMemoryLLM()

    @app.ingest(
        "/ingest",
        embedder=embedder,
        vector_store=vector_store,
    )
    async def ingest_route(request: IngestRequest) -> IngestRequest:
        return request

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
        ingest_response = await client.post(
            "/ingest",
            json={
                "documents": [
                    "FastRAG provides a production-first Python framework for RAG services.",
                    "It is designed to feel familiar to FastAPI users.",
                ],
                "collection": "overview",
                "metadata": {"source_id": "overview", "page_number": 1},
            },
        )
        query_response = await client.post(
            "/query",
            json={
                "query": "What is FastRAG?",
                "collection": "overview",
                "top_k": 2,
            },
        )

    assert ingest_response.status_code == 200
    assert ingest_response.json() == {
        "status": "completed",
        "ingested_documents": 2,
        "collection": "overview",
        "tenant_id": None,
    }
    assert query_response.status_code == 200
    assert query_response.json()["citations"][0]["source_id"] == "overview-0"
    assert "FastRAG" in query_response.json()["answer"]


def test_ingest_decorator_rejects_invalid_components() -> None:
    app = create_app()

    with pytest.raises(TypeError, match="embedder must implement the Embedder protocol"):
        app.ingest(
            "/ingest",
            embedder=cast(Embedder, object()),
            vector_store=InMemoryVectorStore(),
        )
