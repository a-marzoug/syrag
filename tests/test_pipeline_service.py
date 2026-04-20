from collections.abc import Sequence

import pytest

from fastrag.protocols import Chunker, Embedder, EmbeddingVector, PromptAssembler, VectorStore
from fastrag.providers import (
    InMemoryEmbedder,
    InMemoryLLM,
    InMemoryVectorStore,
    PassThroughChunker,
)
from fastrag.schemas import (
    AssembledPrompt,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    RetrievedChunk,
)
from fastrag.services import PipelineService, RetrievalStrategy


class StubIngestionPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[IngestRequest, str, str, str]] = []

    async def run(
        self,
        *,
        request: IngestRequest,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> IngestResponse:
        self.calls.append(
            (
                request,
                type(chunker).__name__,
                type(embedder).__name__,
                type(vector_store).__name__,
            )
        )
        return IngestResponse(
            status="completed",
            ingested_documents=len(request.documents),
            collection=request.collection,
            tenant_id=request.tenant_id,
        )


class StubRetrievalStrategy(RetrievalStrategy):
    def __init__(self) -> None:
        self.calls: list[tuple[QueryRequest, list[float], str]] = []

    async def retrieve(
        self,
        *,
        request: QueryRequest,
        query_embedding: EmbeddingVector,
        vector_store: VectorStore,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            (
                request,
                [float(value) for value in query_embedding],
                type(vector_store).__name__,
            )
        )
        return [
            RetrievedChunk(
                chunk_id="overview-0-chunk-0",
                source_id="overview-0",
                content="FastRAG emphasizes observability and type safety.",
                score=0.99,
                metadata={},
                page_number=1,
                chunk_index=0,
            )
        ]


class StubPromptAssembler(PromptAssembler):
    def __init__(self) -> None:
        self.calls: list[tuple[QueryRequest, list[str]]] = []

    async def assemble(
        self,
        *,
        query: QueryRequest,
        context: Sequence[RetrievedChunk],
    ) -> AssembledPrompt:
        self.calls.append((query, [chunk.source_id for chunk in context]))
        return AssembledPrompt(
            query=query,
            context=[
                RetrievedChunk(
                    chunk_id="assembled-chunk-0",
                    source_id="assembled-doc",
                    content="Prompt assembly can reshape the grounded context.",
                    score=0.88,
                    metadata={},
                    page_number=1,
                    chunk_index=0,
                )
            ],
            prompt=f"Custom assembled prompt for: {query.query}",
        )


@pytest.mark.asyncio
async def test_pipeline_service_runs_ingest_and_query_flow() -> None:
    service = PipelineService()
    embedder = InMemoryEmbedder()
    vector_store = InMemoryVectorStore()
    llm = InMemoryLLM()

    ingest_response = await service.run_ingest(
        request=IngestRequest(
            documents=[
                "FastRAG is a production-first Python framework for RAG services.",
                "It emphasizes observability and type safety.",
            ],
            collection="overview",
            metadata={"source_id": "overview", "page_number": 1},
        ),
        chunker=PassThroughChunker(),
        embedder=embedder,
        vector_store=vector_store,
    )
    query_response = await service.run_query(
        request=QueryRequest(
            query="What does FastRAG emphasize?",
            collection="overview",
            top_k=2,
        ),
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
    )

    assert ingest_response.ingested_documents == 2
    assert query_response.citations[0].source_id == "overview-0"
    assert "FastRAG" in query_response.answer
    assert query_response.usage["prompt_tokens"] > len(
        "What does FastRAG emphasize?".split()
    )


@pytest.mark.asyncio
async def test_pipeline_service_delegates_ingest_to_ingestion_pipeline() -> None:
    ingestion_pipeline = StubIngestionPipeline()
    service = PipelineService(ingestion_pipeline=ingestion_pipeline)
    request = IngestRequest(
        documents=["doc one"],
        collection="docs",
    )
    chunker = PassThroughChunker()
    embedder = InMemoryEmbedder()
    vector_store = InMemoryVectorStore()

    response = await service.run_ingest(
        request=request,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    assert response.ingested_documents == 1
    assert ingestion_pipeline.calls == [
        (request, "PassThroughChunker", "InMemoryEmbedder", "InMemoryVectorStore")
    ]


@pytest.mark.asyncio
async def test_pipeline_service_delegates_query_retrieval_to_retrieval_strategy() -> None:
    retrieval_strategy = StubRetrievalStrategy()
    service = PipelineService(retrieval_strategy=retrieval_strategy)
    embedder = InMemoryEmbedder()
    vector_store = InMemoryVectorStore()
    llm = InMemoryLLM()
    request = QueryRequest(query="What does FastRAG emphasize?", collection="overview")

    response = await service.run_query(
        request=request,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
    )
    expected_embedding = await embedder.embed([request.query])

    assert response.citations[0].source_id == "overview-0"
    assert retrieval_strategy.calls == [
        (request, expected_embedding[0], "InMemoryVectorStore")
    ]


@pytest.mark.asyncio
async def test_pipeline_service_delegates_prompt_assembly_to_prompt_assembler() -> None:
    retrieval_strategy = StubRetrievalStrategy()
    prompt_assembler = StubPromptAssembler()
    service = PipelineService(
        retrieval_strategy=retrieval_strategy,
        prompt_assembler=prompt_assembler,
    )
    embedder = InMemoryEmbedder()
    vector_store = InMemoryVectorStore()
    llm = InMemoryLLM()
    request = QueryRequest(query="What does FastRAG emphasize?", collection="overview")

    response = await service.run_query(
        request=request,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
    )

    assert response.citations[0].source_id == "assembled-doc"
    assert prompt_assembler.calls == [(request, ["overview-0"])]
