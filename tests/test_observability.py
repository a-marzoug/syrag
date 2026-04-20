import pytest
from httpx import ASGITransport, AsyncClient

from fastrag.app import create_app
from fastrag.observability import PipelineEvent
from fastrag.providers import (
    InMemoryEmbedder,
    InMemoryLLM,
    InMemoryVectorStore,
    PassThroughChunker,
)
from fastrag.schemas import DocumentChunk, IngestRequest, QueryRequest


@pytest.mark.asyncio
async def test_query_route_emits_stage_events() -> None:
    app = create_app()
    embedder = InMemoryEmbedder()
    vector_store = InMemoryVectorStore()
    llm = InMemoryLLM()
    events: list[PipelineEvent] = []
    app.add_event_listener(events.append)

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
        )

    assert response.status_code == 200
    assert [(event.stage, event.status) for event in events] == [
        ("embed", "started"),
        ("embed", "succeeded"),
        ("retrieve", "started"),
        ("retrieve", "succeeded"),
        ("assemble", "started"),
        ("assemble", "succeeded"),
        ("generate", "started"),
        ("generate", "succeeded"),
    ]


@pytest.mark.asyncio
async def test_ingest_route_emits_stage_events() -> None:
    app = create_app()
    events: list[PipelineEvent] = []
    app.add_event_listener(events.append)

    @app.ingest(
        "/ingest",
        chunker=PassThroughChunker(),
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
    )
    async def ingest_route(request: IngestRequest) -> IngestRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/ingest", json={"documents": ["FastRAG doc"]})

    assert response.status_code == 200
    assert [(event.stage, event.status) for event in events] == [
        ("chunk", "started"),
        ("chunk", "succeeded"),
        ("embed", "started"),
        ("embed", "succeeded"),
        ("store", "started"),
        ("store", "succeeded"),
    ]
