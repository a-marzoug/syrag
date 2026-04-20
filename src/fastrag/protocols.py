from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from fastrag.schemas import (
    AssembledPrompt,
    DocumentChunk,
    QueryRequest,
    RAGResponse,
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
class LLM(Protocol):
    """Generates the final grounded response from retrieved context."""

    async def generate(
        self,
        *,
        prompt: AssembledPrompt,
    ) -> RAGResponse:
        """Return a typed response with answer text, citations, and usage."""
