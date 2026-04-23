from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from fastrag.app import FastRAG, create_app
from fastrag.config import ComponentDefaults, Settings
from fastrag.protocols import (
    LLM,
    Chunker,
    Embedder,
    EmbeddingVector,
    Filters,
    VectorStore,
)
from fastrag.schemas import (
    Citation,
    DocumentChunk,
    GenerationRequest,
    RAGResponse,
    RetrievedChunk,
    SourceDocument,
)


@dataclass(slots=True)
class EmbedCall:
    texts: list[str]


@dataclass(slots=True)
class ChunkCall:
    documents: list[SourceDocument]


@dataclass(slots=True)
class UpsertCall:
    chunks: list[DocumentChunk]
    embeddings: list[list[float]]
    collection: str | None
    tenant_id: str | None


@dataclass(slots=True)
class QueryCall:
    query_embedding: list[float]
    top_k: int
    collection: str | None
    tenant_id: str | None
    filters: dict[str, Any]


@dataclass(slots=True)
class GenerateCall:
    generation: GenerationRequest


class FakeEmbedder(Embedder):
    """Deterministic fake embedder that records input texts."""

    def __init__(self, *, dimensions: int = 2) -> None:
        self.dimensions = dimensions
        self.calls: list[EmbedCall] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        normalized_texts = [str(text) for text in texts]
        self.calls.append(EmbedCall(texts=normalized_texts))
        return [self._embed_text(text) for text in normalized_texts]

    def _embed_text(self, text: str) -> list[float]:
        length = float(len(text))
        token_count = float(len(text.split()))
        base_vector = [length, token_count]
        if self.dimensions <= 2:
            return base_vector[: self.dimensions]
        return [*base_vector, *([0.0] * (self.dimensions - 2))]


class FakeChunker(Chunker):
    """Simple chunker that emits one retrieval chunk per source document."""

    def __init__(self) -> None:
        self.calls: list[ChunkCall] = []

    async def chunk(
        self,
        documents: Sequence[SourceDocument],
    ) -> list[DocumentChunk]:
        normalized_documents = [document.model_copy(deep=True) for document in documents]
        self.calls.append(ChunkCall(documents=normalized_documents))
        return [
            DocumentChunk(
                chunk_id=f"{document.source_id}-chunk-0",
                source_id=document.source_id,
                content=document.content,
                metadata=dict(document.metadata),
                page_number=document.page_number,
                chunk_index=0,
            )
            for document in normalized_documents
        ]


class FakeVectorStore(VectorStore):
    """In-memory fake vector store for deterministic query and ingest tests."""

    def __init__(
        self,
        *,
        query_results: Sequence[RetrievedChunk] | None = None,
    ) -> None:
        self.query_results = [chunk.model_copy(deep=True) for chunk in query_results or []]
        self.upsert_calls: list[UpsertCall] = []
        self.query_calls: list[QueryCall] = []
        self._stored_chunks: dict[tuple[str | None, str | None], list[RetrievedChunk]] = {}

    async def upsert(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
        collection: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        copied_chunks = [chunk.model_copy(deep=True) for chunk in chunks]
        copied_embeddings = [
            [float(value) for value in embedding]
            for embedding in embeddings
        ]
        self.upsert_calls.append(
            UpsertCall(
                chunks=copied_chunks,
                embeddings=copied_embeddings,
                collection=collection,
                tenant_id=tenant_id,
            )
        )
        namespace = self._stored_chunks.setdefault((collection, tenant_id), [])
        for chunk in copied_chunks:
            namespace.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    content=chunk.content,
                    score=1.0,
                    metadata=dict(chunk.metadata),
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                )
            )

    async def query(
        self,
        *,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: Filters | None = None,
    ) -> list[RetrievedChunk]:
        self.query_calls.append(
            QueryCall(
                query_embedding=[float(value) for value in query_embedding],
                top_k=top_k,
                collection=collection,
                tenant_id=tenant_id,
                filters=dict(filters or {}),
            )
        )
        if self.query_results:
            return [chunk.model_copy(deep=True) for chunk in self.query_results[:top_k]]
        stored_chunks = self._stored_chunks.get((collection, tenant_id), [])
        matching_chunks = [
            chunk
            for chunk in stored_chunks
            if self._matches_filters(chunk.metadata, filters)
        ]
        return [chunk.model_copy(deep=True) for chunk in matching_chunks[:top_k]]

    def _matches_filters(
        self,
        metadata: Mapping[str, Any],
        filters: Filters | None,
    ) -> bool:
        if not filters:
            return True
        return all(metadata.get(key) == value for key, value in filters.items())


class FakeLLM(LLM):
    """Fake generator that records requests and emits grounded answers."""

    def __init__(
        self,
        *,
        answer: str | None = None,
        usage: Mapping[str, int] | None = None,
    ) -> None:
        self.answer = answer
        self.usage = dict(usage or {"prompt_tokens": 0, "completion_tokens": 0})
        self.calls: list[GenerateCall] = []

    async def generate(
        self,
        *,
        generation: GenerationRequest,
    ) -> RAGResponse:
        self.calls.append(GenerateCall(generation=generation.model_copy(deep=True)))
        answer = self.answer or self._default_answer(generation)
        return RAGResponse(
            answer=answer,
            citations=self._citations_for(generation),
            usage=dict(self.usage),
        )

    def _default_answer(self, generation: GenerationRequest) -> str:
        if generation.context:
            joined_context = " ".join(chunk.content for chunk in generation.context)
            return f"Grounded answer for '{generation.query.query}': {joined_context}"
        return f"No grounded context was available for query: {generation.query.query}"

    def _citations_for(self, generation: GenerationRequest) -> list[Citation]:
        if not generation.require_citations:
            return []
        return [
            Citation(
                source_id=chunk.source_id,
                score=chunk.score,
                snippet=chunk.content,
                page_number=chunk.page_number,
            )
            for chunk in generation.context
        ]


@dataclass(slots=True)
class FakeProviderBundle:
    embedder: FakeEmbedder = field(default_factory=FakeEmbedder)
    vector_store: FakeVectorStore = field(default_factory=FakeVectorStore)
    llm: FakeLLM = field(default_factory=FakeLLM)
    chunker: FakeChunker = field(default_factory=FakeChunker)


def create_test_app(
    settings: Settings | None = None,
    *,
    providers: FakeProviderBundle | None = None,
) -> FastRAG:
    """Create a FastRAG app preconfigured with fake providers and defaults."""

    provider_bundle = providers or FakeProviderBundle()
    resolved_settings = settings or Settings()
    app = create_app(
        resolved_settings.model_copy(
            update={
                "defaults": ComponentDefaults(
                    embedder="test",
                    vector_store="test",
                    llm="test",
                )
            },
            deep=True,
        )
    )
    app.register_embedder("test", provider_bundle.embedder)
    app.register_vector_store("test", provider_bundle.vector_store)
    app.register_llm("test", provider_bundle.llm)
    app.chunker = cast(Any, provider_bundle.chunker)
    app.api.state.chunker = provider_bundle.chunker
    return app


def create_test_client(
    app: FastRAG | FastAPI,
    *,
    base_url: str = "http://testserver",
) -> AsyncClient:
    """Create an HTTPX client configured for a FastRAG or FastAPI ASGI app."""

    asgi_app = app.api if isinstance(app, FastRAG) else app
    return AsyncClient(transport=ASGITransport(app=asgi_app), base_url=base_url)


async def seed_documents(
    app: FastRAG,
    *,
    documents: Sequence[str],
    collection: str | None = None,
    tenant_id: str | None = None,
    metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    embedder: Embedder | str | None = None,
    vector_store: VectorStore | str | None = None,
) -> list[DocumentChunk]:
    """Embed and store source text in a target vector store for integration tests."""

    resolved_embedder = app.resolver.resolve_embedder(embedder)
    resolved_vector_store = app.resolver.resolve_vector_store(vector_store)
    chunks = _build_seed_chunks(documents=documents, metadata=metadata)
    embeddings = await resolved_embedder.embed([chunk.content for chunk in chunks])
    await resolved_vector_store.upsert(
        chunks=chunks,
        embeddings=embeddings,
        collection=collection,
        tenant_id=tenant_id,
    )
    return chunks


def _build_seed_chunks(
    *,
    documents: Sequence[str],
    metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[DocumentChunk]:
    metadata_items = _normalize_metadata(documents=documents, metadata=metadata)
    chunks: list[DocumentChunk] = []
    for index, document in enumerate(documents):
        chunk_metadata = dict(metadata_items[index])
        source_id = str(chunk_metadata.pop("source_id", f"doc-{index}"))
        page_number_raw = chunk_metadata.pop("page_number", None)
        page_number = page_number_raw if isinstance(page_number_raw, int) else None
        chunks.append(
            DocumentChunk(
                chunk_id=f"{source_id}-chunk-0",
                source_id=source_id,
                content=document,
                metadata=chunk_metadata,
                page_number=page_number,
                chunk_index=0,
            )
        )
    return chunks


def _normalize_metadata(
    *,
    documents: Sequence[str],
    metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if metadata is None:
        return [{} for _ in documents]
    if isinstance(metadata, Mapping):
        return [dict(metadata) for _ in documents]
    if len(metadata) != len(documents):
        msg = "metadata sequence must match the number of documents"
        raise ValueError(msg)
    return [dict(item) for item in metadata]
