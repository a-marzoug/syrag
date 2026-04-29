from collections.abc import Sequence

import pytest
from starlette.requests import Request

from syrag.protocols import (
    LLM,
    AuthHook,
    Chunker,
    Embedder,
    EmbeddingVector,
    GenerationPolicy,
    PromptAssembler,
    RateLimiter,
    RequestContextHook,
    SafetyGuard,
    VectorStore,
)
from syrag.schemas import (
    AssembledPrompt,
    Citation,
    DocumentChunk,
    GenerationRequest,
    IngestRequest,
    QueryRequest,
    RAGResponse,
    RequestContext,
    RetrievedChunk,
    SourceDocument,
)
from syrag.services import RetrievalStrategy


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
                content="SyRAG wraps FastAPI for RAG workloads.",
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


class ExampleRequestContextHook:
    async def enrich(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> RequestContext:
        return context.model_copy(
            update={"request_id": "request-1", "metadata": {"path": request.url.path}}
        )


class ExampleAuthHook:
    async def authenticate(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> RequestContext:
        return context.model_copy(update={"subject_id": request.headers.get("x-user-id")})


class ExampleLLM:
    async def generate(
        self,
        *,
        generation: GenerationRequest,
    ) -> RAGResponse:
        return RAGResponse(
            answer=f"Answering: {generation.query.query}",
            citations=[
                Citation(
                    source_id=generation.context[0].source_id,
                    score=generation.context[0].score,
                    snippet=generation.context[0].content,
                    page_number=generation.context[0].page_number,
                )
            ],
            usage={"prompt_tokens": 64, "completion_tokens": 16},
        )


class ExampleRateLimiter:
    async def check(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> None:
        del request, context


class ExampleSafetyGuard:
    async def validate_query(
        self,
        *,
        request: Request,
        payload: QueryRequest,
        context: RequestContext,
    ) -> QueryRequest:
        del request, context
        return payload

    async def validate_ingest(
        self,
        *,
        request: Request,
        payload: IngestRequest,
        context: RequestContext,
    ) -> IngestRequest:
        del request, context
        return payload


class ExamplePromptAssembler:
    async def assemble(
        self,
        *,
        query: QueryRequest,
        context: Sequence[RetrievedChunk],
    ) -> AssembledPrompt:
        return AssembledPrompt(
            query=query,
            context=list(context),
            prompt=f"Question: {query.query}",
        )


class ExampleGenerationPolicy:
    async def apply(
        self,
        *,
        prompt: AssembledPrompt,
    ) -> GenerationRequest:
        return GenerationRequest(
            query=prompt.query,
            context=prompt.context,
            prompt=prompt.prompt,
            system_prompt="Ground the answer in context.",
            require_citations=True,
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
    assert isinstance(ExampleRequestContextHook(), RequestContextHook)
    assert isinstance(ExampleAuthHook(), AuthHook)
    assert isinstance(ExampleLLM(), LLM)
    assert isinstance(ExampleRateLimiter(), RateLimiter)
    assert isinstance(ExampleSafetyGuard(), SafetyGuard)
    assert isinstance(ExampleGenerationPolicy(), GenerationPolicy)
    assert isinstance(ExamplePromptAssembler(), PromptAssembler)
    assert isinstance(ExampleRetrievalStrategy(), RetrievalStrategy)


@pytest.mark.asyncio
async def test_example_llm_returns_typed_response() -> None:
    llm = ExampleLLM()
    response = await llm.generate(
        generation=GenerationRequest(
            query=QueryRequest(query="What is SyRAG?"),
            context=[
                RetrievedChunk(
                    chunk_id="prd-chunk-0",
                    source_id="prd",
                    content="SyRAG is a production-first Python framework for RAG services.",
                    score=0.99,
                    metadata={},
                    page_number=1,
                    chunk_index=0,
                )
            ],
            prompt="Question: What is SyRAG?",
            system_prompt="Ground the answer in context.",
            require_citations=True,
        ),
    )

    assert response.citations[0].source_id == "prd"
    assert response.usage["completion_tokens"] == 16
