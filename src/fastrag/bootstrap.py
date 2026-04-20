from __future__ import annotations

from fastrag.config import BootstrapSettings, ComponentDefaults, ProviderSettings
from fastrag.providers import InMemoryProviderFactory, ProviderFactory
from fastrag.registry import ComponentRegistry


class BootstrapService:
    """Applies lightweight app bootstrap behavior from configuration."""

    def __init__(
        self,
        settings: BootstrapSettings,
        *,
        provider_settings: ProviderSettings,
        factory: ProviderFactory | None = None,
    ) -> None:
        self.settings = settings
        self.provider_settings = provider_settings
        self.factory = factory or InMemoryProviderFactory()

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
                self.factory.create_embedder(settings=self.provider_settings),
            )

        if defaults.vector_store is not None:
            registry.register_vector_store(
                defaults.vector_store,
                self.factory.create_vector_store(settings=self.provider_settings),
            )

        if defaults.llm is not None:
            registry.register_llm(
                defaults.llm,
                self.factory.create_llm(settings=self.provider_settings),
            )
