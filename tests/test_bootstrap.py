import pytest
from httpx import ASGITransport, AsyncClient

from fastrag.app import create_app
from fastrag.config import BootstrapSettings, ComponentDefaults, Settings
from fastrag.registry import ComponentNotFoundError
from fastrag.schemas import IngestRequest, QueryRequest


@pytest.mark.asyncio
async def test_bootstrap_can_register_default_in_memory_components() -> None:
    app = create_app(
        Settings(
            defaults=ComponentDefaults(
                embedder="default",
                vector_store="memory",
                llm="grounded",
            ),
            bootstrap=BootstrapSettings(register_in_memory_defaults=True),
        )
    )

    @app.ingest("/ingest")
    async def ingest_route(request: IngestRequest) -> IngestRequest:
        return request

    @app.query("/query")
    async def query_route(request: QueryRequest) -> QueryRequest:
        return request

    async with AsyncClient(
        transport=ASGITransport(app=app.api),
        base_url="http://testserver",
    ) as client:
        ingest_response = await client.post(
            "/ingest",
            json={
                "documents": ["FastRAG bootstraps default in-memory components."],
                "collection": "bootstrap",
                "metadata": {"source_id": "bootstrap-doc", "page_number": 1},
            },
        )
        query_response = await client.post(
            "/query",
            json={"query": "How does bootstrap work?", "collection": "bootstrap"},
        )

    assert ingest_response.status_code == 200
    assert query_response.status_code == 200
    assert query_response.json()["citations"][0]["source_id"] == "bootstrap-doc"


def test_bootstrap_disabled_does_not_register_default_components() -> None:
    app = create_app(
        Settings(
            defaults=ComponentDefaults(
                embedder="default",
                vector_store="memory",
                llm="grounded",
            ),
            bootstrap=BootstrapSettings(register_in_memory_defaults=False),
        )
    )

    with pytest.raises(ComponentNotFoundError, match="embedder 'default' is not registered"):
        app.query("/query")
