from __future__ import annotations

from typing import Final

from fastrag.protocols import LLM, Embedder, VectorStore


class RegistryError(Exception):
    """Base exception for component registry failures."""


class ComponentAlreadyRegisteredError(RegistryError):
    """Raised when a component name is reused within the same role."""


class ComponentNotFoundError(RegistryError):
    """Raised when a named component cannot be resolved."""


class ComponentValidationError(RegistryError):
    """Raised when a component does not satisfy its expected protocol."""


class ComponentRegistry:
    """In-process registry for named FastRAG components."""

    def __init__(self) -> None:
        self._embedders: dict[str, Embedder] = {}
        self._vector_stores: dict[str, VectorStore] = {}
        self._llms: dict[str, LLM] = {}

    def register_embedder(self, name: str, component: Embedder) -> None:
        self._register_component(
            name=name,
            component=component,
            storage=self._embedders,
            protocol=Embedder,
            role_name="embedder",
        )

    def register_vector_store(self, name: str, component: VectorStore) -> None:
        self._register_component(
            name=name,
            component=component,
            storage=self._vector_stores,
            protocol=VectorStore,
            role_name="vector_store",
        )

    def register_llm(self, name: str, component: LLM) -> None:
        self._register_component(
            name=name,
            component=component,
            storage=self._llms,
            protocol=LLM,
            role_name="llm",
        )

    def get_embedder(self, name: str) -> Embedder:
        return self._get_component(name=name, storage=self._embedders, role_name="embedder")

    def get_vector_store(self, name: str) -> VectorStore:
        return self._get_component(
            name=name,
            storage=self._vector_stores,
            role_name="vector_store",
        )

    def get_llm(self, name: str) -> LLM:
        return self._get_component(name=name, storage=self._llms, role_name="llm")

    def _register_component[T](
        self,
        *,
        name: str,
        component: T,
        storage: dict[str, T],
        protocol: type[object],
        role_name: str,
    ) -> None:
        normalized_name = self._normalize_name(name)
        if normalized_name in storage:
            msg = f"{role_name} '{normalized_name}' is already registered"
            raise ComponentAlreadyRegisteredError(msg)
        if not isinstance(component, protocol):
            msg = f"{role_name} '{normalized_name}' must implement the {protocol.__name__} protocol"
            raise ComponentValidationError(msg)
        storage[normalized_name] = component

    def _get_component[T](
        self,
        *,
        name: str,
        storage: dict[str, T],
        role_name: str,
    ) -> T:
        normalized_name = self._normalize_name(name)
        try:
            return storage[normalized_name]
        except KeyError as exc:
            msg = f"{role_name} '{normalized_name}' is not registered"
            raise ComponentNotFoundError(msg) from exc

    def _normalize_name(self, name: str) -> str:
        normalized_name = name.strip().lower()
        if not normalized_name:
            msg = "component name must not be blank"
            raise ValueError(msg)
        return normalized_name


DEFAULT_QUERY_TAGS: Final[list[str]] = ["query"]
DEFAULT_INGEST_TAGS: Final[list[str]] = ["ingest"]
