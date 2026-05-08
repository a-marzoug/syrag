from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from starlette.requests import Request

from syrag.schemas import (
    AssembledPrompt,
    DocumentChunk,
    GenerationRequest,
    IngestRequest,
    QueryRequest,
    RAGResponse,
    RequestContext,
    RetrievedChunk,
    SourceDocument,
)

type EmbeddingVector = Sequence[float]
type Filters = Mapping[str, Any]


@runtime_checkable
class Embedder(Protocol):
    """Generates embeddings for raw text inputs."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


@runtime_checkable
class VectorStore(Protocol):
    """Persists and retrieves embedded documents."""

    async def upsert(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
        collection: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """Insert or update chunks and their embeddings."""

    async def query(
        self,
        *,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: Filters | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query embedding."""


@runtime_checkable
class Chunker(Protocol):
    """Splits source documents into retrieval-ready chunks."""

    async def chunk(
        self,
        documents: Sequence[SourceDocument],
    ) -> list[DocumentChunk]:
        """Return chunked representations derived from source documents."""


@runtime_checkable
class RequestContextHook(Protocol):
    """Builds request-scoped framework context from an incoming HTTP request."""

    async def enrich(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> RequestContext:
        """Return the request context after applying request-scoped metadata."""


@runtime_checkable
class AuthHook(Protocol):
    """Applies authentication/authorization context to a request-scoped context."""

    async def authenticate(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> RequestContext:
        """Return the request context after authentication is applied."""


@runtime_checkable
class RateLimiter(Protocol):
    """Applies request-level throttling before route handlers execute."""

    async def check(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> None:
        """Raise when the incoming request exceeds the configured rate limit."""


@runtime_checkable
class SafetyGuard(Protocol):
    """Validates request payloads before they reach pipeline execution."""

    async def validate_query(
        self,
        *,
        request: Request,
        payload: QueryRequest,
        context: RequestContext,
    ) -> QueryRequest:
        """Return a validated query payload or raise when it is unsafe."""

    async def validate_ingest(
        self,
        *,
        request: Request,
        payload: IngestRequest,
        context: RequestContext,
    ) -> IngestRequest:
        """Return a validated ingest payload or raise when it is unsafe."""


@runtime_checkable
class PromptAssembler(Protocol):
    """Builds a grounded prompt package from query inputs and retrieved chunks."""

    async def assemble(
        self,
        *,
        query: QueryRequest,
        context: Sequence[RetrievedChunk],
    ) -> AssembledPrompt:
        """Return the assembled prompt payload passed into generation."""


@runtime_checkable
class Reranker(Protocol):
    """Reorders or filters retrieved chunks before prompt assembly."""

    async def rerank(
        self,
        *,
        query: QueryRequest,
        context: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Return reranked or filtered chunks for the query."""


@runtime_checkable
class GenerationPolicy(Protocol):
    """Applies generation-time constraints and instructions to an assembled prompt."""

    async def apply(
        self,
        *,
        prompt: AssembledPrompt,
    ) -> GenerationRequest:
        """Return the final generation request passed into the LLM."""


@runtime_checkable
class LLM(Protocol):
    """Generates the final grounded response from retrieved context."""

    async def generate(
        self,
        *,
        generation: GenerationRequest,
    ) -> RAGResponse:
        """Return a typed response with answer text, citations, and usage."""
