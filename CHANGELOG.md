# Changelog

All notable changes to SyRAG will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added optional FAISS vector store provider behind the `faiss` extra.
- Added optional Google Gen AI embedder and LLM providers behind the `google` extra.
- Added optional LangChain text splitter adapter behind the `langchain` extra.
- Added optional LangChain retriever strategy adapter behind the `langchain` extra.
- Added optional SyRAG query LangChain tool factory behind the `langchain` extra.
- Added optional LlamaIndex node parser adapter behind the `llamaindex` extra.
- Added optional LlamaIndex retriever strategy adapter behind the `llamaindex` extra.
- Added optional SyRAG query engine adapter for LlamaIndex behind the `llamaindex` extra.
- Added the `Reranker` protocol for retrieval post-processing.

## [0.1.0] - 2026-04-30

### Added

- Added the `SyRAG` application wrapper on top of FastAPI.
- Added typed ingest and query decorators.
- Added typed request, response, document, chunk, citation, and request-context schemas.
- Added protocol-based extension points for chunking, embedding, vector storage, retrieval, prompt assembly, generation policy, LLM generation, request context, auth, rate limiting, and safety validation.
- Added in-memory providers for local development and tests.
- Added optional Chroma vector store provider behind the `chroma` extra.
- Added `SQLiteVectorStore` for lightweight persistent vector storage.
- Added optional OpenAI embedder and LLM providers behind the `openai` extra.
- Added tenant-aware request context and tenant normalization.
- Added structured SyRAG error responses with stage and category metadata.
- Added OpenTelemetry-compatible tracing integration.
- Added structured logging and JSON log formatting.
- Added default rate limiting and safety guard implementations.
- Added OpenAPI examples and documented framework route responses.
- Added a downstream testing toolkit with fake providers and ASGI client helpers.
- Added provider contract tests across first-party provider implementations.
- Added cookbook documentation for Qdrant, LangChain, LangGraph, and LlamaIndex RAG integration patterns.
- Added package extras for `chroma`, `openai`, `testing`, `server`, and `all`.
- Added a minimal OpenAI and Chroma-backed application example.
- Added release, contribution, security, and community documentation for open-source use.
- Added CI for linting, type checking, tests, package building, and package metadata checks.
- Added tag-driven PyPI publishing workflow through PyPI Trusted Publishing.

### Changed

- Renamed the public package, import namespace, CLI, and framework branding to `syrag` / `SyRAG` before the first public release.
- Set supported Python versions to `3.12` and `3.13`.
- Updated first-time-user documentation to use OpenAI-backed generation and embeddings with a local vector store.

### Notes

- This is the first planned public release.
