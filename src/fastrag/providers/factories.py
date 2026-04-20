from __future__ import annotations

from typing import Protocol

from fastrag.config import ProviderSettings
from fastrag.protocols import LLM, Embedder, VectorStore
from fastrag.providers.in_memory import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore


class ProviderFactory(Protocol):
    """Factory contract for bootstrap-managed component construction."""

    def create_embedder(self, *, settings: ProviderSettings) -> Embedder: ...

    def create_vector_store(self, *, settings: ProviderSettings) -> VectorStore: ...

    def create_llm(self, *, settings: ProviderSettings) -> LLM: ...


class InMemoryProviderFactory:
    """Default factory for local in-memory framework components."""

    def create_embedder(self, *, settings: ProviderSettings) -> Embedder:
        return InMemoryEmbedder(dimensions=settings.in_memory.embedder_dimensions)

    def create_vector_store(self, *, settings: ProviderSettings) -> VectorStore:
        return InMemoryVectorStore()

    def create_llm(self, *, settings: ProviderSettings) -> LLM:
        return InMemoryLLM(
            max_context_documents=settings.in_memory.llm_max_context_documents
        )
