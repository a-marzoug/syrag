from syrag.app import create_app
from syrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from syrag.schemas import IngestRequest, QueryRequest


def test_query_route_openapi_includes_examples_and_error_responses() -> None:
    app = create_app()

    @app.query(
        "/query",
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        llm=InMemoryLLM(),
    )
    async def query_route(request: QueryRequest) -> QueryRequest:
        """Application-specific query behavior."""
        return request

    schema = app.api.openapi()
    operation = schema["paths"]["/query"]["post"]

    assert operation["summary"] == "Run a grounded query"
    assert "Application-specific query behavior." in operation["description"]
    assert (
        operation["requestBody"]["content"]["application/json"]["examples"]["grounded_query"][
            "value"
        ]["collection"]
        == "overview"
    )
    assert operation["responses"]["200"]["headers"]["x-request-id"]["schema"]["type"] == "string"
    assert (
        operation["responses"]["200"]["content"]["application/json"]["examples"][
            "grounded_answer"
        ]["value"]["citations"][0]["source_id"]
        == "overview"
    )
    assert (
        operation["responses"]["400"]["description"]
        == "Request validation or safety guard failure."
    )
    assert operation["responses"]["429"]["headers"]["retry-after"]["schema"]["type"] == "string"
    assert operation["responses"]["500"]["content"]["application/json"]["examples"][
        "generation_failed"
    ]["value"]["error"]["stage"] == "generate"


def test_ingest_and_health_routes_openapi_include_examples() -> None:
    app = create_app()

    @app.ingest(
        "/ingest",
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
    )
    async def ingest_route(request: IngestRequest) -> IngestRequest:
        return request

    schema = app.api.openapi()
    ingest_operation = schema["paths"]["/ingest"]["post"]
    health_operation = schema["paths"]["/health"]["get"]

    assert ingest_operation["summary"] == "Ingest source documents"
    assert (
        ingest_operation["requestBody"]["content"]["application/json"]["examples"][
            "document_batch"
        ]["value"]["documents"][0]
        == "SyRAG is a production-first Python framework for RAG services."
    )
    assert ingest_operation["responses"]["200"]["content"]["application/json"]["examples"][
        "ingest_completed"
    ]["value"]["status"] == "completed"
    assert (
        ingest_operation["responses"]["429"]["headers"]["retry-after"]["schema"]["type"]
        == "string"
    )
    assert health_operation["summary"] == "Health check"
    assert (
        health_operation["responses"]["200"]["content"]["application/json"]["examples"][
            "healthy"
        ]["value"]["status"]
        == "ok"
    )
