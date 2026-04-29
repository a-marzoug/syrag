import pytest

from syrag.schemas import IngestRequest, QueryRequest, RetrievedChunk
from syrag.testing import (
    FakeLLM,
    FakeProviderBundle,
    FakeVectorStore,
    create_test_app,
    create_test_client,
    seed_documents,
)


@pytest.mark.asyncio
async def test_create_test_app_and_seed_documents_support_query_integration_tests() -> None:
    app = create_test_app()
    await seed_documents(
        app,
        documents=["SyRAG is a production-first Python framework for RAG services."],
        collection="overview",
        metadata={"source_id": "overview", "page_number": 1, "topic": "product"},
    )

    @app.query("/query")
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with create_test_client(app) as client:
        response = await client.post(
            "/query",
            json={
                "query": "What is SyRAG?",
                "collection": "overview",
                "filters": {"topic": "product"},
            },
        )

    assert response.status_code == 200
    assert response.json()["citations"][0]["source_id"] == "overview"
    assert "SyRAG is a production-first Python framework" in response.json()["answer"]


@pytest.mark.asyncio
async def test_fake_providers_record_calls_for_query_tests() -> None:
    providers = FakeProviderBundle(
        vector_store=FakeVectorStore(
            query_results=[
                RetrievedChunk(
                    chunk_id="overview-chunk-0",
                    source_id="overview",
                    content="SyRAG wraps FastAPI for RAG applications.",
                    score=0.99,
                    metadata={"topic": "product"},
                    page_number=1,
                    chunk_index=0,
                )
            ]
        ),
        llm=FakeLLM(answer="Custom fake answer."),
    )
    app = create_test_app(providers=providers)

    @app.query("/query")
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with create_test_client(app) as client:
        response = await client.post(
            "/query",
            json={"query": "What does SyRAG wrap?", "collection": "overview", "top_k": 1},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Custom fake answer."
    assert providers.embedder.calls[0].texts == ["What does SyRAG wrap?"]
    assert providers.vector_store.query_calls[0].collection == "overview"
    assert providers.vector_store.query_calls[0].top_k == 1
    assert providers.llm.calls[0].generation.query.query == "What does SyRAG wrap?"


@pytest.mark.asyncio
async def test_create_test_app_supports_ingest_routes_with_fake_chunker() -> None:
    providers = FakeProviderBundle()
    app = create_test_app(providers=providers)

    @app.ingest("/ingest")
    async def ingest_route(request: IngestRequest) -> IngestRequest:
        return request

    async with create_test_client(app) as client:
        response = await client.post(
            "/ingest",
            json={
                "documents": [
                    "SyRAG exposes query decorators.",
                    "SyRAG exposes ingest decorators.",
                ],
                "collection": "overview",
                "metadata": {"source_id": "overview", "page_number": 1},
            },
        )

    assert response.status_code == 200
    assert response.json()["ingested_documents"] == 2
    assert providers.chunker.calls[0].documents[0].source_id == "overview-0"
    assert providers.vector_store.upsert_calls[0].collection == "overview"


@pytest.mark.asyncio
async def test_seed_documents_rejects_metadata_length_mismatches() -> None:
    app = create_test_app()

    with pytest.raises(ValueError, match="metadata sequence must match the number of documents"):
        await seed_documents(
            app,
            documents=["doc-1", "doc-2"],
            metadata=[{"source_id": "doc-1"}],
        )
