import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from syrag.protocols import LLM, Embedder, VectorStore
from syrag.providers import (
    ChromaVectorStore,
    FAISSVectorStore,
    GoogleEmbedder,
    GoogleLLM,
    InMemoryEmbedder,
    InMemoryLLM,
    InMemoryVectorStore,
    OpenAIEmbedder,
    OpenAILLM,
    SQLiteVectorStore,
)
from syrag.schemas import DocumentChunk, GenerationRequest, QueryRequest, RetrievedChunk

type AsyncFactory[T] = Callable[[], Awaitable[T]]


class FakeChromaCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]],
        documents: list[str],
    ) -> None:
        for record_id, embedding, metadata, document in zip(
            ids,
            embeddings,
            metadatas,
            documents,
            strict=True,
        ):
            self.records[record_id] = {
                "embedding": embedding,
                "metadata": metadata,
                "document": document,
            }

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, object],
        include: list[str],
    ) -> dict[str, object]:
        del query_embeddings, include
        matching_records = [
            (record_id, record)
            for record_id, record in self.records.items()
            if self._matches_where(record["metadata"], where)
        ][:n_results]
        return {
            "ids": [[record_id for record_id, _record in matching_records]],
            "documents": [[str(record["document"]) for _record_id, record in matching_records]],
            "metadatas": [[record["metadata"] for _record_id, record in matching_records]],
            "distances": [[0.0 for _record_id, _record in matching_records]],
        }

    def _matches_where(self, metadata: object, where: dict[str, object]) -> bool:
        if not isinstance(metadata, dict):
            return False
        predicates = where.get("$and")
        if isinstance(predicates, list):
            return all(
                isinstance(predicate, dict)
                and all(metadata.get(key) == value for key, value in predicate.items())
                for predicate in predicates
            )
        return all(metadata.get(key) == value for key, value in where.items())


class FakeChromaClient:
    def __init__(self) -> None:
        self.collection = FakeChromaCollection()

    def get_or_create_collection(
        self,
        *,
        name: str,
        embedding_function: object | None,
    ) -> FakeChromaCollection:
        del name
        assert embedding_function is None
        return self.collection


class FakeGoogleModels:
    async def embed_content(
        self,
        *,
        model: str,
        contents: list[str],
        config: object | None,
    ) -> SimpleNamespace:
        del model, config
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[float(index + 1), float(len(text)), 0.5])
                for index, text in enumerate(contents)
            ]
        )

    async def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: object | None,
    ) -> SimpleNamespace:
        del model, contents, config
        return SimpleNamespace(
            text="SyRAG contract answer.",
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=4,
                total_token_count=16,
            ),
        )


class FakeGoogleClient:
    def __init__(self) -> None:
        self.aio = SimpleNamespace(models=FakeGoogleModels())


async def _build_in_memory_embedder() -> Embedder:
    return InMemoryEmbedder(dimensions=8)


async def _build_openai_embedder() -> Embedder:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        raw_inputs = payload.get("input", [])
        if not isinstance(raw_inputs, list):
            raw_inputs = []
        data = [
            {
                "index": index,
                "embedding": [float(index + 1), float(len(str(text))), 0.5],
            }
            for index, text in enumerate(raw_inputs)
        ]
        return httpx.Response(status_code=200, json={"data": data})

    return OpenAIEmbedder(
        api_key="test-key",
        model="text-embedding-3-small",
        transport=httpx.MockTransport(handler),
    )


async def _build_google_embedder() -> Embedder:
    return GoogleEmbedder(client=FakeGoogleClient(), model="gemini-embedding-001")


async def _build_in_memory_vector_store() -> VectorStore:
    return InMemoryVectorStore()


async def _build_sqlite_vector_store(tmp_path: Path) -> VectorStore:
    return SQLiteVectorStore(tmp_path / "provider-contracts.sqlite3")


async def _build_chroma_vector_store() -> VectorStore:
    return ChromaVectorStore(client=FakeChromaClient())


async def _build_faiss_vector_store() -> VectorStore:
    return FAISSVectorStore(dimensions=2)


async def _build_in_memory_llm() -> LLM:
    return InMemoryLLM()


async def _build_openai_llm() -> LLM:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": "SyRAG contract answer."}
                        ]
                    }
                ],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "total_tokens": 16,
                },
            },
        )

    return OpenAILLM(
        api_key="test-key",
        model="gpt-5.4-mini",
        transport=httpx.MockTransport(handler),
    )


async def _build_google_llm() -> LLM:
    return GoogleLLM(client=FakeGoogleClient(), model="gemini-2.5-flash")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "factory"),
    [
        ("in_memory", _build_in_memory_embedder),
        ("openai", _build_openai_embedder),
        ("google", _build_google_embedder),
    ],
)
async def test_embedder_contract_returns_one_float_vector_per_input(
    provider_name: str,
    factory: AsyncFactory[Embedder],
) -> None:
    embedder = await factory()

    embeddings = await embedder.embed(["SyRAG", "RAG services"])
    empty_embeddings = await embedder.embed([])

    assert isinstance(embedder, Embedder), provider_name
    assert len(embeddings) == 2, provider_name
    assert empty_embeddings == [], provider_name
    assert all(isinstance(vector, list) for vector in embeddings), provider_name
    assert all(vector for vector in embeddings), provider_name
    assert all(isinstance(value, float) for vector in embeddings for value in vector), provider_name


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["in_memory", "sqlite", "chroma", "faiss"])
async def test_vector_store_contract_supports_namespace_filters_and_upsert_replacement(
    provider_name: str,
    tmp_path: Path,
) -> None:
    if provider_name == "in_memory":
        store = await _build_in_memory_vector_store()
    elif provider_name == "sqlite":
        store = await _build_sqlite_vector_store(tmp_path)
    elif provider_name == "chroma":
        store = await _build_chroma_vector_store()
    else:
        store = await _build_faiss_vector_store()

    await store.upsert(
        chunks=[
            DocumentChunk(
                chunk_id="alpha-chunk-0",
                source_id="alpha",
                content="Original alpha content.",
                metadata={"topic": "alpha"},
                page_number=1,
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="beta-chunk-0",
                source_id="beta",
                content="Beta content.",
                metadata={"topic": "beta"},
                page_number=2,
                chunk_index=0,
            ),
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        collection="overview",
        tenant_id="tenant-a",
    )
    await store.upsert(
        chunks=[
            DocumentChunk(
                chunk_id="alpha-chunk-0",
                source_id="alpha",
                content="Updated alpha content.",
                metadata={"topic": "alpha", "updated": True},
                page_number=3,
                chunk_index=0,
            )
        ],
        embeddings=[[1.0, 0.0]],
        collection="overview",
        tenant_id="tenant-a",
    )
    await store.upsert(
        chunks=[
            DocumentChunk(
                chunk_id="tenant-b-chunk-0",
                source_id="tenant-b",
                content="Other tenant content.",
                metadata={"topic": "alpha"},
                page_number=1,
                chunk_index=0,
            )
        ],
        embeddings=[[1.0, 0.0]],
        collection="overview",
        tenant_id="tenant-b",
    )

    results = await store.query(
        query_embedding=[1.0, 0.0],
        top_k=2,
        collection="overview",
        tenant_id="tenant-a",
        filters={"topic": "alpha"},
    )
    other_tenant_results = await store.query(
        query_embedding=[1.0, 0.0],
        top_k=2,
        collection="overview",
        tenant_id="tenant-b",
    )

    assert isinstance(store, VectorStore), provider_name
    assert len(results) == 1, provider_name
    assert results[0].source_id == "alpha", provider_name
    assert results[0].content == "Updated alpha content.", provider_name
    assert results[0].page_number == 3, provider_name
    assert 0.0 <= results[0].score <= 1.0, provider_name
    assert len(other_tenant_results) == 1, provider_name
    assert other_tenant_results[0].source_id == "tenant-b", provider_name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "factory"),
    [
        ("in_memory", _build_in_memory_llm),
        ("openai", _build_openai_llm),
        ("google", _build_google_llm),
    ],
)
async def test_llm_contract_returns_answer_usage_and_optional_citations(
    provider_name: str,
    factory: AsyncFactory[LLM],
) -> None:
    llm = await factory()
    generation = GenerationRequest(
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
        system_prompt="Ground the answer in context.",
        require_citations=True,
    )
    generation_without_citations = generation.model_copy(update={"require_citations": False})

    response = await llm.generate(generation=generation)
    response_without_citations = await llm.generate(generation=generation_without_citations)

    assert isinstance(llm, LLM), provider_name
    assert isinstance(response.answer, str) and response.answer, provider_name
    assert response.citations[0].source_id == "overview", provider_name
    assert isinstance(response.usage, dict), provider_name
    assert response_without_citations.citations == [], provider_name
