from collections.abc import Sequence

import pytest

from fastrag.protocols import LLM, Chunker, Embedder, EmbeddingVector, VectorStore
from fastrag.schemas import (
    Citation,
    DocumentChunk,
    QueryRequest,
    RAGResponse,
    RetrievedChunk,
    SourceDocument,
)
from fastrag.services import RetrievalStrategy


class ExampleEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


class ExampleVectorStore:
    async def upsert(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
        collection: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        return None

    async def query(
        self,
        *,
        query_embedding: Sequence[float],
        top_k: int,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="doc-1-chunk-0",
                source_id="doc-1",
                content="FastRAG wraps FastAPI for RAG workloads.",
                score=0.95,
                metadata={},
                page_number=1,
                chunk_index=0,
            )
        ]


class ExampleChunker:
    async def chunk(
        self,
        documents: Sequence[SourceDocument],
    ) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_id=f"{document.source_id}-chunk-0",
                source_id=document.source_id,
                content=document.content,
                metadata=document.metadata,
                page_number=document.page_number,
                chunk_index=0,
            )
            for document in documents
        ]


class ExampleLLM:
    async def generate(
        self,
        *,
        query: QueryRequest,
        context: Sequence[RetrievedChunk],
    ) -> RAGResponse:
        return RAGResponse(
            answer=f"Answering: {query.query}",
            citations=[
                Citation(
                    source_id=context[0].source_id,
                    score=context[0].score,
                    snippet=context[0].content,
                    page_number=context[0].page_number,
                )
            ],
            usage={"prompt_tokens": 64, "completion_tokens": 16},
        )


class ExampleRetrievalStrategy:
    async def retrieve(
        self,
        *,
        request: QueryRequest,
        query_embedding: EmbeddingVector,
        vector_store: VectorStore,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="doc-1-chunk-0",
                source_id="doc-1",
                content=f"Retrieved for: {request.query}",
                score=0.95,
                metadata={},
                page_number=1,
                chunk_index=0,
            )
        ]


def test_example_components_match_runtime_protocols() -> None:
    assert isinstance(ExampleEmbedder(), Embedder)
    assert isinstance(ExampleVectorStore(), VectorStore)
    assert isinstance(ExampleChunker(), Chunker)
    assert isinstance(ExampleLLM(), LLM)
    assert isinstance(ExampleRetrievalStrategy(), RetrievalStrategy)


@pytest.mark.asyncio
async def test_example_llm_returns_typed_response() -> None:
    llm = ExampleLLM()
    response = await llm.generate(
        query=QueryRequest(query="What is FastRAG?"),
        context=[
            RetrievedChunk(
                chunk_id="prd-chunk-0",
                source_id="prd",
                content="FastRAG is a production-first Python framework for RAG services.",
                score=0.99,
                metadata={},
                page_number=1,
                chunk_index=0,
            )
        ],
    )

    assert response.citations[0].source_id == "prd"
    assert response.usage["completion_tokens"] == 16
