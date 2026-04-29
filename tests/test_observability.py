import pytest
from httpx import ASGITransport, AsyncClient

from syrag.app import create_app
from syrag.observability import PipelineEvent
from syrag.providers import (
    InMemoryEmbedder,
    InMemoryLLM,
    InMemoryVectorStore,
    PassThroughChunker,
)
from syrag.schemas import DocumentChunk, IngestRequest, QueryRequest


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
        )

    assert response.status_code == 200
    assert [(event.stage, event.status) for event in events] == [
        ("embed", "started"),
        ("embed", "succeeded"),
        ("retrieve", "started"),
        ("retrieve", "succeeded"),
        ("assemble", "started"),
        ("assemble", "succeeded"),
        ("policy", "started"),
        ("policy", "succeeded"),
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
        response = await client.post("/ingest", json={"documents": ["SyRAG doc"]})

    assert response.status_code == 200
    assert [(event.stage, event.status) for event in events] == [
        ("chunk", "started"),
        ("chunk", "succeeded"),
        ("embed", "started"),
        ("embed", "succeeded"),
        ("store", "started"),
        ("store", "succeeded"),
    ]
