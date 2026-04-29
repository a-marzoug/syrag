import pytest
from httpx import ASGITransport, AsyncClient

from syrag.app import create_app
from syrag.config import ComponentDefaults, Settings
from syrag.errors import DependencyConfigurationError
from syrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from syrag.schemas import IngestRequest, QueryRequest


@pytest.mark.asyncio
async def test_routes_can_use_configured_default_component_names() -> None:
    app = create_app(
        Settings(
            defaults=ComponentDefaults(
                embedder="default",
                vector_store="memory",
                llm="grounded",
            )
        )
    )
    app.register_embedder("default", InMemoryEmbedder())
    app.register_vector_store("memory", InMemoryVectorStore())
    app.register_llm("grounded", InMemoryLLM())

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
                "documents": ["SyRAG can resolve registered defaults automatically."],
                "collection": "defaults",
                "metadata": {"source_id": "defaults-doc", "page_number": 1},
            },
        )
        query_response = await client.post(
            "/query",
            json={"query": "How does SyRAG resolve defaults?", "collection": "defaults"},
        )

    assert ingest_response.status_code == 200
    assert query_response.status_code == 200
    assert query_response.json()["citations"][0]["source_id"] == "defaults-doc"


def test_missing_default_component_configuration_is_rejected() -> None:
    app = create_app(Settings(defaults=ComponentDefaults()))

    with pytest.raises(
        DependencyConfigurationError,
        match="No default embedder configured for this app",
    ):
        app.query(
            "/query",
            vector_store=InMemoryVectorStore(),
            llm=InMemoryLLM(),
        )


def test_app_defaults_can_be_updated_programmatically() -> None:
    app = create_app(Settings(defaults=ComponentDefaults()))

    app.configure_defaults(embedder="default", vector_store="memory", llm="grounded")

    assert app.defaults == ComponentDefaults(
        embedder="default",
        vector_store="memory",
        llm="grounded",
    )
