from collections.abc import Sequence
from typing import cast

import pytest
from httpx import ASGITransport, AsyncClient

from fastrag.app import create_app
from fastrag.protocols import Embedder, EmbeddingVector, PromptAssembler, VectorStore
from fastrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from fastrag.schemas import AssembledPrompt, DocumentChunk, QueryRequest, RetrievedChunk
from fastrag.services import RetrievalStrategy


class StubRetrievalStrategy(RetrievalStrategy):
    async def retrieve(
        self,
        *,
        request: QueryRequest,
        query_embedding: EmbeddingVector,
        vector_store: VectorStore,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="custom-chunk-0",
                source_id="custom-doc",
                content=f"Custom retrieval for: {request.query}",
                score=0.87,
                metadata={},
                page_number=1,
                chunk_index=0,
            )
        ]


class StubPromptAssembler(PromptAssembler):
    async def assemble(
        self,
        *,
        query: QueryRequest,
        context: Sequence[RetrievedChunk],
    ) -> AssembledPrompt:
        return AssembledPrompt(
            query=query,
            context=[
                RetrievedChunk(
                    chunk_id="assembled-chunk-0",
                    source_id="assembled-doc",
                    content=f"Custom assembly for: {query.query}",
                    score=0.91,
                    metadata={},
                    page_number=1,
                    chunk_index=0,
                )
            ],
            prompt=f"Custom assembled prompt for: {query.query}",
        )


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


def test_query_decorator_rejects_invalid_retrieval_strategies() -> None:
    app = create_app()

    with pytest.raises(
        TypeError,
        match="retrieval_strategy must implement the RetrievalStrategy protocol",
    ):
        app.query(
            "/query",
            embedder=InMemoryEmbedder(),
            vector_store=InMemoryVectorStore(),
            llm=InMemoryLLM(),
            retrieval_strategy=cast(RetrievalStrategy, object()),
        )


def test_query_decorator_rejects_invalid_prompt_assemblers() -> None:
    app = create_app()

    with pytest.raises(
        TypeError,
        match="prompt_assembler must implement the PromptAssembler protocol",
    ):
        app.query(
            "/query",
            embedder=InMemoryEmbedder(),
            vector_store=InMemoryVectorStore(),
            llm=InMemoryLLM(),
            prompt_assembler=cast(PromptAssembler, object()),
        )


@pytest.mark.asyncio
async def test_query_decorator_accepts_custom_retrieval_strategy() -> None:
    app = create_app()

    @app.query(
        "/query",
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        llm=InMemoryLLM(),
        retrieval_strategy=StubRetrievalStrategy(),
    )
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/query", json={"query": "What is FastRAG?"})

    assert response.status_code == 200
    assert response.json()["citations"][0]["source_id"] == "custom-doc"
    assert "Custom retrieval" in response.json()["answer"]


@pytest.mark.asyncio
async def test_query_decorator_accepts_custom_prompt_assembler() -> None:
    app = create_app()

    @app.query(
        "/query",
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        llm=InMemoryLLM(),
        prompt_assembler=StubPromptAssembler(),
    )
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/query", json={"query": "What is FastRAG?"})

    assert response.status_code == 200
    assert response.json()["citations"][0]["source_id"] == "assembled-doc"
    assert "Custom assembly" in response.json()["answer"]
