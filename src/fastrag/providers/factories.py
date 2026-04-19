from __future__ import annotations

from typing import Protocol

from fastrag.config import BootstrapSettings
from fastrag.protocols import LLM, Embedder, VectorStore
from fastrag.providers.in_memory import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore


class ProviderFactory(Protocol):
    """Factory contract for bootstrap-managed component construction."""

    def create_embedder(self, *, settings: BootstrapSettings) -> Embedder: ...

    def create_vector_store(self, *, settings: BootstrapSettings) -> VectorStore: ...

    def create_llm(self, *, settings: BootstrapSettings) -> LLM: ...


class InMemoryProviderFactory:
    """Default factory for local in-memory framework components."""

    def create_embedder(self, *, settings: BootstrapSettings) -> Embedder:
        return InMemoryEmbedder(dimensions=settings.in_memory_embedder_dimensions)

    def create_vector_store(self, *, settings: BootstrapSettings) -> VectorStore:
        return InMemoryVectorStore()

    def create_llm(self, *, settings: BootstrapSettings) -> LLM:
        return InMemoryLLM(
            max_context_documents=settings.in_memory_llm_max_context_documents
        )
