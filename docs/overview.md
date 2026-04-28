# Product Overview

## What FastRAG Is

FastRAG is a Python framework for defining, serving, and operating RAG services with a small application surface and explicit pipeline boundaries. It is built around typed HTTP routes, protocol-based components, and production-facing defaults.

The current promise is straightforward:

- define ingest and query routes with a FastAPI-like API
- swap providers and pipeline stages behind stable protocols
- keep request context, tenant scoping, observability, and safety in the framework surface

## Problem It Solves

Teams building RAG systems usually start with notebooks or provider SDK calls, then spend significant time assembling the same service concerns around them. That usually creates three problems:

- too much boilerplate between prototype and production
- inconsistent interfaces across embedders, vector stores, and generators
- operational concerns such as request identity, multi-tenancy, and error reporting getting bolted on late

FastRAG addresses that by making the RAG service, not the raw provider call, the unit of application design.

## Who It Serves

Primary users:

- backend and ML engineers shipping RAG-backed APIs
- teams that want typed interfaces and observability without adopting a large orchestration platform

Secondary users:

- teams that need a clean framework seam before adding external providers or custom policies
- application teams that want a testable local path with in-memory components

## Product Pillars

### 1. Fast developer experience

FastRAG lets developers stand up a usable service with typed schemas, decorators, and generated OpenAPI docs.

### 2. Protocol-first design

Core stages and hooks are defined as runtime-checkable protocols so applications can replace them without changing route shape.

### 3. Production-first defaults

Request context, tenant handling, guardrails, structured errors, structured logging, and tracing are framework concepts rather than ad hoc application code.

### 4. Low lock-in

The core package stays small while provider integrations and testing helpers remain behind explicit extras when appropriate.

## What Ships Today

Application surface:

- `FastRAG(...)` and `create_app(...)`
- `@app.ingest(...)` and `@app.query(...)`
- `QueryRequest`, `IngestRequest`, `RAGResponse`, and related schemas
- `/health` plus OpenAPI docs through FastAPI

Pipeline surface:

- chunking during ingest
- embedding and vector store persistence
- retrieval strategy abstraction for query execution
- prompt assembly and generation policy seams before the LLM
- typed final responses with citations and usage metadata

Operational surface:

- request-scoped `RequestContext`
- auth and request-context hooks
- tenant normalization from headers and payloads
- in-process rate limiting and safety guards
- OpenTelemetry-compatible tracing
- structured logging
- structured FastRAG error taxonomy

Provider surface:

- in-memory embedder, vector store, and LLM
- SQLite-backed vector store
- OpenAI embedder and LLM behind the `openai` extra

Testing surface:

- fake providers
- ASGI test client helper
- document seeding helper
- provider contract tests across first-party implementations

## Current Boundaries

FastRAG currently does not try to be:

- a general agent orchestration system
- a plugin marketplace
- a streaming query framework
- a full evaluation platform
- a batteries-included metrics exporter

Those can be added later, but they are not documented as current features.

## Minimal Shape

The common path looks like this:

```python
from fastrag import (
    FastRAG,
    InMemoryEmbedder,
    InMemoryLLM,
    InMemoryVectorStore,
    IngestRequest,
    QueryRequest,
    Settings,
)

app = FastRAG(
    title="Support Bot",
    version="0.1.0",
    description="Internal support assistant",
    settings=Settings(),
)

app.register_embedder("default", InMemoryEmbedder())
app.register_vector_store("default", InMemoryVectorStore())
app.register_llm("default", InMemoryLLM())
app.configure_defaults(embedder="default", vector_store="default", llm="default")


@app.ingest("/ingest")
async def ingest(request: IngestRequest) -> IngestRequest:
    return request


@app.query("/query")
async def query(request: QueryRequest) -> QueryRequest:
    return request
```
