# SyRAG 0.3.0 Roadmap

The proposed `0.3.0` theme is provider maturity and interoperability hardening.

SyRAG should remain a production service framework, not a replacement for mature RAG strategy ecosystems. The next release should make the existing integration story more reliable, better tested, and easier to trust in real applications before expanding into many new abstractions.

## Primary Scope

- Add a first-party Qdrant vector store provider behind a `qdrant` optional extra.
- Add provider contract tests for Qdrant using local or in-memory Qdrant modes where practical.
- Strengthen LangChain adapter tests around retriever inputs, tool outputs, metadata preservation, and error handling.
- Strengthen LlamaIndex adapter tests around node parsing, retriever responses, query-engine behavior, metadata preservation, and error handling.
- Document public API stability levels for core schemas, protocols, providers, integrations, and testing helpers.
- Clarify which providers are production-oriented and which are development/test utilities.
- Add a release checklist that covers docs publishing, PyPI verification, package smoke tests, and post-release validation.

## Supporting Scope

- Add compatibility notes for Qdrant and any new optional dependencies.
- Add cookbook coverage for a complete OpenAI + Qdrant SyRAG service.
- Add cookbook coverage for using Qdrant through LangChain or LlamaIndex while keeping SyRAG as the service boundary.
- Improve package smoke tests so each optional extra has at least one import-level and construction-level check.
- Keep CI focused on deterministic checks that do not require live provider credentials.

## Design Checks

- Core imports must not require Qdrant, LangChain, LlamaIndex, OpenAI, Google, Chroma, or FAISS.
- Optional extras should install only the dependencies needed for that integration.
- Provider adapters should preserve document IDs, text, scores, metadata, collection names, and tenant-related filtering where supported.
- Integration adapters should wrap existing ecosystem objects rather than reimplementing their retrieval or chunking strategies.
- Error handling should continue to normalize failures into SyRAG stage-aware errors at framework boundaries.
- Public API additions should be documented before they are treated as stable.

## Deferred

- Semantic Kernel adapters.
- Haystack adapters.
- DSPy adapters.
- CrewAI adapters.
- Streaming query responses.
- Evaluation framework and benchmark runner.
- Hosted metrics exporter configuration.
- Broad first-party strategy implementations for advanced chunking, retrieval, reranking, and agent workflows.

## Open Decisions

- Whether Qdrant should support both local embedded mode and remote hosted mode in the first provider release.
- Whether Qdrant metadata filters should expose a SyRAG-neutral filter shape or accept provider-native filters.
- Whether public APIs should be labeled as `stable`, `provisional`, and `internal` in docs only or also in code docstrings.
- Whether `all` should include future production vector-store extras such as `qdrant`, or whether those should remain explicit installs.
- Whether SyRAG needs a general `PostProcessor` protocol beyond the existing reranker seam.
