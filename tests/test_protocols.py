from collections.abc import Sequence

import pytest

from fastrag.protocols import LLM, Embedder, VectorStore
from fastrag.schemas import Citation, QueryRequest, RAGResponse, RetrievedDocument


class ExampleEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


class ExampleVectorStore:
    async def upsert(
        self,
        *,
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        collection: str | None = None,
        tenant_id: str | None = None,
        metadata: Sequence[dict[str, object]] | None = None,
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
    ) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                source_id="doc-1",
                content="FastRAG wraps FastAPI for RAG workloads.",
                score=0.95,
                metadata={},
                page_number=1,
            )
        ]


class ExampleLLM:
    async def generate(
        self,
        *,
        query: QueryRequest,
        context: Sequence[RetrievedDocument],
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


def test_example_components_match_runtime_protocols() -> None:
    assert isinstance(ExampleEmbedder(), Embedder)
    assert isinstance(ExampleVectorStore(), VectorStore)
    assert isinstance(ExampleLLM(), LLM)


@pytest.mark.asyncio
async def test_example_llm_returns_typed_response() -> None:
    llm = ExampleLLM()
    response = await llm.generate(
        query=QueryRequest(query="What is FastRAG?"),
        context=[
            RetrievedDocument(
                source_id="prd",
                content="FastRAG is a production-first Python framework for RAG services.",
                score=0.99,
                metadata={},
                page_number=1,
            )
        ],
    )

    assert response.citations[0].source_id == "prd"
    assert response.usage["completion_tokens"] == 16
