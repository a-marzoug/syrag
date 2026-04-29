from __future__ import annotations

from syrag.config import ComponentDefaults
from syrag.errors import DependencyConfigurationError
from syrag.protocols import LLM, Embedder, VectorStore
from syrag.registry import ComponentRegistry

type EmbedderRef = Embedder | str | None
type VectorStoreRef = VectorStore | str | None
type LLMRef = LLM | str | None


class ComponentResolver:
    """Resolve concrete components from direct instances, names, and configured defaults."""

    def __init__(
        self,
        *,
        registry: ComponentRegistry,
        defaults: ComponentDefaults,
    ) -> None:
        self._registry = registry
        self._defaults = defaults.model_copy(deep=True)

    def update_defaults(self, defaults: ComponentDefaults) -> None:
        self._defaults = defaults.model_copy(deep=True)

    def resolve_embedder(self, component: EmbedderRef) -> Embedder:
        if component is None:
            default_name = self._require_default_name(
                component_name="embedder",
                configured_name=self._defaults.embedder,
            )
            return self._registry.get_embedder(default_name)
        if isinstance(component, str):
            return self._registry.get_embedder(component)
        self._validate_component(component, Embedder, "embedder")
        return component

    def resolve_vector_store(self, component: VectorStoreRef) -> VectorStore:
        if component is None:
            default_name = self._require_default_name(
                component_name="vector_store",
                configured_name=self._defaults.vector_store,
            )
            return self._registry.get_vector_store(default_name)
        if isinstance(component, str):
            return self._registry.get_vector_store(component)
        self._validate_component(component, VectorStore, "vector_store")
        return component

    def resolve_llm(self, component: LLMRef) -> LLM:
        if component is None:
            default_name = self._require_default_name(
                component_name="llm",
                configured_name=self._defaults.llm,
            )
            return self._registry.get_llm(default_name)
        if isinstance(component, str):
            return self._registry.get_llm(component)
        self._validate_component(component, LLM, "llm")
        return component

    @staticmethod
    def _require_default_name(
        *,
        component_name: str,
        configured_name: str | None,
    ) -> str:
        if configured_name is not None:
            return configured_name

        raise DependencyConfigurationError(
            code="missing_default_component",
            message=f"No default {component_name} configured for this app",
            details={"component": component_name},
        )

    @staticmethod
    def _validate_component(
        component: object,
        protocol: type[object],
        component_name: str,
    ) -> None:
        if isinstance(component, protocol):
            return

        msg = f"{component_name} must implement the {protocol.__name__} protocol"
        raise TypeError(msg)
