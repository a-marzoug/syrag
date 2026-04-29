from typing import cast

import pytest
from httpx import ASGITransport, AsyncClient

from syrag.app import create_app
from syrag.protocols import Embedder
from syrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from syrag.registry import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
    ComponentValidationError,
)
from syrag.schemas import IngestRequest, QueryRequest


def test_registry_rejects_duplicate_component_names() -> None:
    app = create_app()
    embedder = InMemoryEmbedder()

    app.register_embedder("default", embedder)

    with pytest.raises(
        ComponentAlreadyRegisteredError,
        match="embedder 'default' is already registered",
    ):
        app.register_embedder("default", embedder)


def test_registry_raises_for_unknown_named_component() -> None:
    app = create_app()

    with pytest.raises(
        ComponentNotFoundError,
        match="embedder 'missing' is not registered",
    ):
        app.query(
            "/query",
            embedder="missing",
            vector_store=InMemoryVectorStore(),
            llm=InMemoryLLM(),
        )


@pytest.mark.asyncio
async def test_routes_can_resolve_components_from_registry() -> None:
    app = create_app()
    embedder = InMemoryEmbedder()
    vector_store = InMemoryVectorStore()
    llm = InMemoryLLM()

    app.register_embedder("default", embedder)
    app.register_vector_store("memory", vector_store)
    app.register_llm("grounded", llm)

    @app.ingest(
        "/ingest",
        embedder="default",
        vector_store="memory",
    )
    async def ingest_route(request: IngestRequest) -> IngestRequest:
        return request

    @app.query(
        "/query",
        embedder="default",
        vector_store="memory",
        llm="grounded",
    )
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        ingest_response = await client.post(
            "/ingest",
            json={
                "documents": ["SyRAG uses a registry for named component lookup."],
                "collection": "registry",
                "metadata": {"source_id": "registry-doc", "page_number": 1},
            },
        )
        query_response = await client.post(
            "/query",
            json={
                "query": "How does SyRAG resolve components?",
                "collection": "registry",
            },
        )

    assert ingest_response.status_code == 200
    assert query_response.status_code == 200
    assert query_response.json()["citations"][0]["source_id"] == "registry-doc"


def test_registry_rejects_invalid_component_registration() -> None:
    app = create_app()

    with pytest.raises(
        ComponentValidationError,
        match="embedder 'invalid' must implement the Embedder protocol",
    ):
        app.register_embedder("invalid", cast(Embedder, object()))
