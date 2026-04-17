import pytest

from fastrag.protocols import LLM, Embedder, VectorStore
from fastrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from fastrag.schemas import QueryRequest


@pytest.mark.asyncio
async def test_in_memory_embedder_is_deterministic() -> None:
    embedder = InMemoryEmbedder(dimensions=8)

    first = await embedder.embed(["FastRAG builds RAG services quickly."])
    second = await embedder.embed(["FastRAG builds RAG services quickly."])

    assert isinstance(embedder, Embedder)
    assert first == second
    assert len(first[0]) == 8


@pytest.mark.asyncio
async def test_in_memory_vector_store_respects_collection_tenant_and_filters() -> None:
    embedder = InMemoryEmbedder()
    store = InMemoryVectorStore()

    documents = [
        "FastRAG is a production-first Python framework for RAG services.",
        "FastAPI powers the HTTP layer for FastRAG.",
    ]
    embeddings = await embedder.embed(documents)
    await store.upsert(
        documents=documents,
        embeddings=embeddings,
        collection="product",
        tenant_id="tenant-a",
        metadata=[
            {"source_id": "prd", "topic": "product", "page_number": 1},
            {"source_id": "http", "topic": "transport", "page_number": 2},
        ],
    )

    query_embedding = (await embedder.embed(["production-first python framework"]))[0]
    results = await store.query(
        query_embedding=query_embedding,
        top_k=2,
        collection="product",
        tenant_id="tenant-a",
        filters={"topic": "product"},
    )

    assert isinstance(store, VectorStore)
    assert len(results) == 1
    assert results[0].source_id == "prd"
    assert results[0].score > 0.0


@pytest.mark.asyncio
async def test_in_memory_llm_generates_grounded_response_with_citations() -> None:
    embedder = InMemoryEmbedder()
    store = InMemoryVectorStore()
    llm = InMemoryLLM()

    documents = [
        "FastRAG offers a FastAPI-like experience for production RAG services.",
        "It emphasizes observability, type safety, and multi-tenancy.",
    ]
    embeddings = await embedder.embed(documents)
    await store.upsert(
        documents=documents,
        embeddings=embeddings,
        collection="overview",
        metadata=[
            {"source_id": "overview-1", "page_number": 1},
            {"source_id": "overview-2", "page_number": 1},
        ],
    )

    query = QueryRequest(query="What does FastRAG emphasize?", collection="overview")
    query_embedding = (await embedder.embed([query.query]))[0]
    context = await store.query(
        query_embedding=query_embedding,
        top_k=2,
        collection=query.collection,
    )
    response = await llm.generate(query=query, context=context)

    assert isinstance(llm, LLM)
    assert "FastRAG" in response.answer
    assert len(response.citations) == 2
    assert response.citations[0].source_id == "overview-1"
    assert response.usage["prompt_tokens"] > 0
