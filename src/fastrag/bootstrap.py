from __future__ import annotations

from fastrag.config import BootstrapSettings, ComponentDefaults
from fastrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from fastrag.registry import ComponentRegistry


class BootstrapService:
    """Applies lightweight app bootstrap behavior from configuration."""

    def __init__(self, settings: BootstrapSettings) -> None:
        self.settings = settings

    def apply(
        self,
        *,
        registry: ComponentRegistry,
        defaults: ComponentDefaults,
    ) -> None:
        if not self.settings.register_in_memory_defaults:
            return

        if defaults.embedder is not None:
            registry.register_embedder(
                defaults.embedder,
                InMemoryEmbedder(dimensions=self.settings.in_memory_embedder_dimensions),
            )

        if defaults.vector_store is not None:
            registry.register_vector_store(defaults.vector_store, InMemoryVectorStore())

        if defaults.llm is not None:
            registry.register_llm(
                defaults.llm,
                InMemoryLLM(
                    max_context_documents=self.settings.in_memory_llm_max_context_documents
                ),
            )
