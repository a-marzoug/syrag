import pytest
from httpx import ASGITransport, AsyncClient

from fastrag.app import create_app
from fastrag.config import BootstrapSettings, ComponentDefaults, Settings
from fastrag.protocols import LLM, Embedder, VectorStore
from fastrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from fastrag.registry import ComponentNotFoundError
from fastrag.schemas import IngestRequest, QueryRequest


class StubProviderFactory:
    def __init__(self) -> None:
        self.embedder = InMemoryEmbedder(dimensions=7)
        self.vector_store = InMemoryVectorStore()
        self.llm = InMemoryLLM(max_context_documents=2)
        self.calls: list[str] = []

    def create_embedder(self, *, settings: BootstrapSettings) -> Embedder:
        self.calls.append(f"embedder:{settings.in_memory_embedder_dimensions}")
        return self.embedder

    def create_vector_store(self, *, settings: BootstrapSettings) -> VectorStore:
        self.calls.append("vector_store")
        return self.vector_store

    def create_llm(self, *, settings: BootstrapSettings) -> LLM:
        self.calls.append(f"llm:{settings.in_memory_llm_max_context_documents}")
        return self.llm


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


def test_bootstrap_can_use_a_custom_provider_factory() -> None:
    factory = StubProviderFactory()
    app = create_app(
        Settings(
            defaults=ComponentDefaults(
                embedder="default",
                vector_store="memory",
                llm="grounded",
            ),
            bootstrap=BootstrapSettings(
                register_in_memory_defaults=True,
                in_memory_embedder_dimensions=99,
                in_memory_llm_max_context_documents=5,
            ),
        ),
        provider_factory=factory,
    )

    assert app.registry.get_embedder("default") is factory.embedder
    assert app.registry.get_vector_store("memory") is factory.vector_store
    assert app.registry.get_llm("grounded") is factory.llm
    assert app.api.state.provider_factory is factory
    assert factory.calls == ["embedder:99", "vector_store", "llm:5"]
