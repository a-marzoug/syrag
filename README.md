# SyRAG

SyRAG is a production-oriented Python framework for building Retrieval-Augmented Generation services with a small, typed API on top of FastAPI.

Supported Python versions: `3.12` and `3.13`.

It currently ships:

- `SyRAG` application wrapper plus `create_app()`
- `@app.ingest(...)` and `@app.query(...)` decorators
- typed request and response schemas
- in-memory providers for development and tests
- Chroma and FAISS vector stores behind optional extras
- SQLite vector store in core
- OpenAI and Google model providers behind optional extras
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
pip install "syrag[faiss]"
pip install "syrag[google]"
pip install "syrag[langchain]"
pip install "syrag[openai]"
pip install "syrag[testing]"
pip install "syrag[server]"
pip install "syrag[all]"
```

## Quick Start

Install the runtime integrations used in this example:

```bash
pip install "syrag[chroma,openai,server]"
```

Set `OPENAI_API_KEY` in your environment before starting the app.

Create `main.py`:

```python
import os
from pathlib import Path

from syrag import (
    SyRAG,
    ChromaVectorStore,
    IngestRequest,
    OpenAIEmbedder,
    OpenAILLM,
    QueryRequest,
    Settings,
)

app = SyRAG(
    title="Support Bot",
    version="0.1.0",
    description="Internal support assistant",
    settings=Settings(),
)

SUPPORT_COLLECTION = "support"

embedder = OpenAIEmbedder(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small",
)
vector_store = ChromaVectorStore(
    path=Path(".syrag/chroma"),
    collection_name="support_docs",
)
llm = OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4.1-mini")


@app.ingest("/ingest", embedder=embedder, vector_store=vector_store)
async def ingest(request: IngestRequest) -> IngestRequest:
    """Normalize incoming documents before they enter the ingest pipeline."""
    return request.model_copy(
        update={
            "collection": request.collection or SUPPORT_COLLECTION,
            "metadata": {"source": "api", **request.metadata},
        }
    )


@app.query("/query", embedder=embedder, vector_store=vector_store, llm=llm)
async def query(request: QueryRequest) -> QueryRequest:
    """Apply route-level retrieval defaults before generation."""
    return request.model_copy(
        update={
            "collection": request.collection or SUPPORT_COLLECTION,
            "top_k": min(request.top_k, 5),
        }
    )
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

`InMemoryEmbedder`, `InMemoryVectorStore`, and `InMemoryLLM` are development/test utilities. They are not intended as production embedding, retrieval, or generation backends.

Optional `chroma` extra:

- `ChromaVectorStore`

Optional `faiss` extra:

- `FAISSVectorStore`

Optional `google` extra:

- `GoogleEmbedder`
- `GoogleLLM`

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
- [Public API](docs/public-api.md)
- [Compatibility matrix](docs/compatibility.md)
- [Cookbook](docs/cookbook/index.md)
- [Provider examples](docs/provider-examples.md)
- [0.2.0 roadmap](docs/roadmap-0.2.md)
- [Releasing](docs/releasing.md)
- [MVP status](docs/mvp-roadmap.md)

## Examples

- [Minimal app](examples/minimal_app.py)

## Cookbook

- [Qdrant vector store pipeline](docs/cookbook/qdrant.md)
- [LangChain + Qdrant RAG](docs/cookbook/langchain.md)
- [LangChain agent with SyRAG tool](examples/integrations/langchain_syrag_agent.py)
- [LangGraph + SyRAG RAG workflow](docs/cookbook/langgraph.md)
- [LlamaIndex + Qdrant RAG](docs/cookbook/llamaindex.md)

Full example scripts live in [examples/integrations](examples/integrations).
