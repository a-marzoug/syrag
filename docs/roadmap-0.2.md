# SyRAG 0.2.0 Roadmap

The proposed `0.2.0` theme is interoperability.

SyRAG should stay focused on the production service boundary: typed routes, request context, tenant handling, observability, guardrails, provider contracts, and deployment ergonomics. Existing RAG frameworks should provide strategy implementations where they are already mature.

## Primary Scope

- Add `syrag[langchain]`. Done for text splitter adapters.
- Add `syrag[llamaindex]`. Done for node parser adapters.
- Add LangChain text splitter adapters for SyRAG `Chunker`. Done.
- Add LangChain retriever adapters for SyRAG `RetrievalStrategy`. Done.
- Add a SyRAG query API wrapper that agents can call as a LangChain tool. Done.
- Add LlamaIndex node parser adapters for SyRAG `Chunker`. Done.
- Add LlamaIndex retriever adapters for SyRAG `RetrievalStrategy`. Done.
- Add a SyRAG query API wrapper for LlamaIndex query/tool use cases. Done.
- Add `RerankingRetrievalStrategy` after the reranker seam is defined. Done.

## Supporting Scope

- Add optional-extra import smoke tests in CI.
- Add wheel install/import smoke tests in CI.
- Update the compatibility matrix for new integration extras.
- Add cookbook examples showing external strategies plugged into SyRAG ingest and query routes.

## Design Checks

- Adapters should wrap installed framework objects rather than copy strategy logic.
- Core imports must not require LangChain or LlamaIndex.
- Adapter errors should still normalize into SyRAG stage-aware errors where they cross the pipeline boundary.
- Route handlers should not need to change when users swap native SyRAG providers for framework adapters.

## Deferred

- Semantic Kernel adapters.
- Haystack adapters.
- DSPy adapters.
- CrewAI adapters.
- Broad first-party reimplementation of chunking, retrieval, reranking, and agentic RAG algorithms.

## Open Decisions

- Whether adapters should live under `syrag.integrations.*` or `syrag.providers.*`.
- Whether SyRAG also needs a broader `PostProcessor` protocol after `Reranker` and `RerankingRetrievalStrategy`.
