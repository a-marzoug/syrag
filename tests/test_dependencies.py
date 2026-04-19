from typing import cast

import pytest

from fastrag.config import ComponentDefaults
from fastrag.dependencies import ComponentResolver
from fastrag.protocols import Embedder
from fastrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from fastrag.registry import ComponentRegistry


def test_component_resolver_accepts_explicit_component_instances() -> None:
    resolver = ComponentResolver(
        registry=ComponentRegistry(),
        defaults=ComponentDefaults(),
    )
    embedder = InMemoryEmbedder()

    resolved_embedder = resolver.resolve_embedder(embedder)

    assert resolved_embedder is embedder


def test_component_resolver_uses_registered_defaults() -> None:
    registry = ComponentRegistry()
    embedder = InMemoryEmbedder()
    vector_store = InMemoryVectorStore()
    llm = InMemoryLLM()
    registry.register_embedder("default", embedder)
    registry.register_vector_store("memory", vector_store)
    registry.register_llm("grounded", llm)
    resolver = ComponentResolver(
        registry=registry,
        defaults=ComponentDefaults(
            embedder="default",
            vector_store="memory",
            llm="grounded",
        ),
    )

    assert resolver.resolve_embedder(None) is embedder
    assert resolver.resolve_vector_store(None) is vector_store
    assert resolver.resolve_llm(None) is llm


def test_component_resolver_rejects_missing_default_configuration() -> None:
    resolver = ComponentResolver(
        registry=ComponentRegistry(),
        defaults=ComponentDefaults(),
    )

    with pytest.raises(ValueError, match="No default embedder configured for this app"):
        resolver.resolve_embedder(None)


def test_component_resolver_rejects_invalid_component_instances() -> None:
    resolver = ComponentResolver(
        registry=ComponentRegistry(),
        defaults=ComponentDefaults(),
    )

    with pytest.raises(TypeError, match="embedder must implement the Embedder protocol"):
        resolver.resolve_embedder(cast(Embedder, object()))


def test_component_resolver_uses_updated_defaults() -> None:
    registry = ComponentRegistry()
    first_embedder = InMemoryEmbedder(dimensions=8)
    second_embedder = InMemoryEmbedder(dimensions=16)
    registry.register_embedder("first", first_embedder)
    registry.register_embedder("second", second_embedder)
    resolver = ComponentResolver(
        registry=registry,
        defaults=ComponentDefaults(embedder="first"),
    )

    resolver.update_defaults(ComponentDefaults(embedder="second"))

    assert resolver.resolve_embedder(None) is second_embedder
