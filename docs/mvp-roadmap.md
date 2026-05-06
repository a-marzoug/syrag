# MVP Status

This document now tracks the implemented MVP surface rather than a future release plan.

## Implemented Surface

### Application surface

- `SyRAG` application wrapper
- `create_app()` bootstrap helper
- `@app.ingest(...)` and `@app.query(...)`
- typed request, response, chunk, citation, and context schemas
- `/health` route and OpenAPI examples for framework endpoints

### Runtime pipeline

- explicit ingest pipeline with source documents and chunks
- retrieval strategy abstraction
- prompt assembly abstraction
- generation policy abstraction
- async query and ingest execution
- structured SyRAG error responses

### Request-scoped operations

- framework-managed `RequestContext`
- auth and request-context hooks
- tenant-aware request normalization
- rate limiting
- safety guard validation

### First-party providers

- `InMemoryEmbedder` for development and tests
- `InMemoryVectorStore` for development and tests
- `InMemoryLLM` for development and tests
- `PassThroughChunker`
- `ChromaVectorStore`
- `FAISSVectorStore`
- `SQLiteVectorStore`
- `GoogleEmbedder` and `GoogleLLM` behind the `google` extra
- `OpenAIEmbedder` and `OpenAILLM` behind the `openai` extra

### Observability and testing

- OpenTelemetry-compatible tracing
- structured logging and JSON log formatting
- provider contract tests
- testing toolkit with fake providers and ASGI test client helpers

### Packaging

- optional extras for OpenAI, testing, and server tooling
- import guards that keep optional dependencies out of the core import path

## Deliberate Omissions

The MVP currently does not include:

- streaming query responses
- reranker or post-processor protocols
- plugin discovery
- built-in metrics export
- background job orchestration
- broad provider coverage
- evaluation workflows
- enterprise RBAC or audit systems

## Why This Scope Works

The current implementation proves the intended framework shape:

- the service surface is small
- the runtime path is explicit and testable
- provider boundaries are real rather than hardcoded
- operational concerns are present without forcing external infrastructure

## Likely Next Expansion Areas

Reasonable next areas, if the project expands beyond the MVP, are:

- LangChain and LlamaIndex strategy adapters
- reranker or post-processor protocols if adapter work requires them
- streaming generation
- metrics export
- contributor and extension author guides

## Planned 0.2.0 Direction

The recommended next release theme is interoperability. SyRAG should remain the typed service, routing, observability, tenancy, and guardrail layer, while mature RAG frameworks provide strategy implementations where they already exist.

Target 0.2.0 scope:

- `syrag[langchain]` for LangChain text splitters, retrievers, and agent tools. Text splitter, retriever, and SyRAG query tool adapters are implemented.
- `syrag[llamaindex]` for LlamaIndex node parsers, retrievers, and query/tool wrappers
- optional-extra import smoke tests in CI
- cookbook examples showing external strategies plugged into SyRAG routes
- a reranker/post-processor protocol only if the integration adapters show the current retrieval seam is too narrow

Deferred beyond 0.2.0:

- Semantic Kernel adapters
- Haystack adapters
- DSPy adapters
- CrewAI adapters
- broad first-party reimplementation of chunking and retrieval algorithms

## Decision Filter

Future work should clear most of these checks:

- does it reduce time-to-production for real RAG services?
- does it strengthen the protocol-first model?
- does it improve production readiness without adding heavy lock-in?
- can it ship without bloating the core package?
- can an existing ecosystem strategy be adapted instead of reimplemented?
