# SyRAG

SyRAG is a production-oriented Python framework for building Retrieval-Augmented Generation services with a small, typed API on top of FastAPI.

Supported Python versions: `3.12` and `3.13`.

It currently ships:

- `SyRAG` application wrapper plus `create_app()`
- `@app.ingest(...)` and `@app.query(...)` decorators
- typed request and response schemas
- in-memory providers for local development
- Chroma vector store behind the `chroma` extra
- SQLite vector store in core and OpenAI providers behind the `openai` extra
- request context, auth hooks, tenant scoping, rate limiting, and safety guards
- OpenAPI docs, structured logging, and OpenTelemetry-compatible tracing
- a testing toolkit with fake providers and ASGI client helpers

## Installation

Core package:

```bash
pip install syrag
```

Optional integrations:

```bash
pip install "syrag[chroma]"
pip install "syrag[openai]"
pip install "syrag[testing]"
pip install "syrag[server]"
pip install "syrag[all]"
```

## Quick Start

Create `main.py`:

```python
from syrag import (
    SyRAG,
    InMemoryEmbedder,
    InMemoryLLM,
    InMemoryVectorStore,
    IngestRequest,
    QueryRequest,
    Settings,
)

app = SyRAG(
    title="Support Bot",
    version="0.1.0",
    description="Internal support assistant",
    settings=Settings(),
)

app.register_embedder("default", InMemoryEmbedder())
app.register_vector_store("default", InMemoryVectorStore())
app.register_llm("default", InMemoryLLM())
app.configure_defaults(
    embedder="default",
    vector_store="default",
    llm="default",
)


@app.ingest("/ingest")
async def ingest(request: IngestRequest) -> IngestRequest:
    return request


@app.query("/query")
async def query(request: QueryRequest) -> QueryRequest:
    return request
```

Serve the app with any ASGI server. With the `server` extra installed:

```bash
uvicorn main:app.api --reload
```

The framework exposes:

- `POST /ingest`
- `POST /query`
- `GET /health`
- OpenAPI docs at `/docs`

Ingest a document:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "content-type: application/json" \
  -d '{"documents":["SyRAG builds typed RAG services."],"collection":"demo"}'
```

Query it:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "content-type: application/json" \
  -d '{"query":"What does SyRAG build?","collection":"demo","top_k":1}'
```

## Extension Points

SyRAG keeps the pipeline explicit. You can swap or extend:

- `Chunker`
- `Embedder`
- `VectorStore`
- `RetrievalStrategy`
- `PromptAssembler`
- `GenerationPolicy`
- `LLM`
- `RequestContextHook`
- `AuthHook`
- `RateLimiter`
- `SafetyGuard`

## First-Party Providers

Core:

- `InMemoryEmbedder`
- `InMemoryVectorStore`
- `InMemoryLLM`
- `PassThroughChunker`
- `SQLiteVectorStore`

Optional `chroma` extra:

- `ChromaVectorStore`

Optional `openai` extra:

- `OpenAIEmbedder`
- `OpenAILLM`

## Observability And Operations

SyRAG includes:

- structured error responses with stage information
- request-scoped `RequestContext` with request IDs and tenant IDs
- `StructuredLogging` and `JSONLogFormatter`
- `OpenTelemetryTracing` built on the OpenTelemetry API package
- request throttling via `InMemoryRateLimiter`
- payload validation via `DefaultSafetyGuard`

## Testing

Install the `testing` extra to use:

- `create_test_app(...)`
- `create_test_client(...)`
- `seed_documents(...)`
- fake providers such as `FakeEmbedder`, `FakeVectorStore`, and `FakeLLM`

## Docs

- [Docs index](docs/index.md)
- [Overview](docs/overview.md)
- [Architecture](docs/architecture.md)
- [Component contracts](docs/component-contracts.md)
- [Cookbook](docs/cookbook/index.md)
- [Provider examples](docs/provider-examples.md)
- [Releasing](docs/releasing.md)
- [MVP status](docs/mvp-roadmap.md)

## Examples

- [Minimal app](examples/minimal_app.py)

## Cookbook

- [Qdrant vector store pipeline](docs/cookbook/qdrant.md)
- [LangChain + Qdrant RAG](docs/cookbook/langchain.md)
- [LangGraph + SyRAG RAG workflow](docs/cookbook/langgraph.md)
- [LlamaIndex + Qdrant RAG](docs/cookbook/llamaindex.md)

Full example scripts live in [examples/integrations](examples/integrations).
