import json
from pathlib import Path

import httpx
import pytest

from syrag.errors import ProviderRequestError, ProviderResponseError
from syrag.providers import OpenAIEmbedder, OpenAILLM, SQLiteVectorStore
from syrag.schemas import DocumentChunk, GenerationRequest, QueryRequest, RetrievedChunk


@pytest.mark.asyncio
async def test_sqlite_vector_store_persists_across_instances(tmp_path: Path) -> None:
    database_path = tmp_path / "vector-store.sqlite3"
    store = SQLiteVectorStore(database_path)
    await store.upsert(
        chunks=[
            DocumentChunk(
                chunk_id="product-chunk-0",
                source_id="product",
                content="SyRAG is production-first.",
                metadata={"topic": "product"},
                page_number=1,
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="http-chunk-0",
                source_id="http",
                content="FastAPI powers the HTTP layer.",
                metadata={"topic": "transport"},
                page_number=1,
                chunk_index=0,
            ),
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        collection="overview",
        tenant_id="tenant-a",
    )

    reopened_store = SQLiteVectorStore(database_path)
    results = await reopened_store.query(
        query_embedding=[1.0, 0.0],
        top_k=2,
        collection="overview",
        tenant_id="tenant-a",
        filters={"topic": "product"},
    )
    missing_tenant_results = await reopened_store.query(
        query_embedding=[1.0, 0.0],
        top_k=2,
        collection="overview",
        tenant_id="tenant-b",
    )

    assert len(results) == 1
    assert results[0].source_id == "product"
    assert results[0].score == pytest.approx(1.0)
    assert missing_tenant_results == []


@pytest.mark.asyncio
async def test_openai_embedder_calls_embeddings_api() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload == {
            "model": "text-embedding-3-small",
            "input": ["hello", "world"],
            "encoding_format": "float",
            "dimensions": 3,
        }
        return httpx.Response(
            status_code=200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                ]
            },
        )

    provider = OpenAIEmbedder(
        api_key="test-key",
        model="text-embedding-3-small",
        dimensions=3,
        transport=httpx.MockTransport(handler),
    )

    assert await provider.embed(["hello", "world"]) == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


@pytest.mark.asyncio
async def test_openai_llm_calls_responses_api_and_maps_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "gpt-5.4-mini"
        assert payload["input"] == [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "Ground answers in context."}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Question: What is SyRAG?"}],
            },
        ]
        return httpx.Response(
            status_code=200,
            json={
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": "SyRAG is a Python RAG framework."}
                        ]
                    }
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )

    provider = OpenAILLM(
        api_key="test-key",
        model="gpt-5.4-mini",
        transport=httpx.MockTransport(handler),
    )

    response = await provider.generate(
        generation=GenerationRequest(
            query=QueryRequest(query="What is SyRAG?"),
            context=[
                RetrievedChunk(
                    chunk_id="overview-chunk-0",
                    source_id="overview",
                    content="SyRAG is a production-first Python framework for RAG services.",
                    score=0.98,
                    metadata={},
                    page_number=1,
                    chunk_index=0,
                )
            ],
            prompt="Question: What is SyRAG?",
            system_prompt="Ground answers in context.",
            require_citations=True,
        )
    )

    assert response.answer == "SyRAG is a Python RAG framework."
    assert response.citations[0].source_id == "overview"
    assert response.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }


@pytest.mark.asyncio
async def test_openai_embedder_raises_provider_request_error_for_http_failures() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=503, json={"error": {"message": "unavailable"}})

    provider = OpenAIEmbedder(
        api_key="test-key",
        model="text-embedding-3-small",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(
        ProviderRequestError,
        match="OpenAI embeddings request failed with status 503",
    ):
        await provider.embed(["hello"])


@pytest.mark.asyncio
async def test_openai_llm_raises_provider_response_error_for_missing_output_text() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json={"output": []})

    provider = OpenAILLM(
        api_key="test-key",
        model="gpt-5.4-mini",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(
        ProviderResponseError,
        match="OpenAI responses output did not contain text output",
    ):
        await provider.generate(
            generation=GenerationRequest(
                query=QueryRequest(query="What is SyRAG?"),
                context=[],
                prompt="Question: What is SyRAG?",
                require_citations=False,
            )
        )
