# MVP Status

This document now tracks the implemented MVP surface rather than a future release plan.

## Implemented Surface

### Application surface

- `FastRAG` application wrapper
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
- structured FastRAG error responses

### Request-scoped operations

- framework-managed `RequestContext`
- auth and request-context hooks
- tenant-aware request normalization
- rate limiting
- safety guard validation

### First-party providers

- `InMemoryEmbedder`
- `InMemoryVectorStore`
- `InMemoryLLM`
- `PassThroughChunker`
- `SQLiteVectorStore`
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

- more specialized vector-store providers
- richer retrieval strategies and reranking
- streaming generation
- metrics export
- contributor and extension author guides

## Decision Filter

Future work should clear most of these checks:

- does it reduce time-to-production for real RAG services?
- does it strengthen the protocol-first model?
- does it improve production readiness without adding heavy lock-in?
- can it ship without bloating the core package?
