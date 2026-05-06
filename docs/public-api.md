# Public API

SyRAG exposes a convenience API from the top-level `syrag` package. This is the preferred import path for application code.

## Primary Application Surface

These names are intended for normal application use:

- `SyRAG`
- `create_app`
- `Settings`
- `IngestRequest`
- `IngestResponse`
- `QueryRequest`
- `RAGResponse`
- `Citation`
- `SourceDocument`
- `DocumentChunk`
- `RetrievedChunk`
- `RequestContext`
- `__version__`

## Extension Protocols

Provider and policy implementations should depend on protocols rather than concrete framework internals:

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

## First-Party Components

These components are available as first-party utilities:

- `PassThroughChunker`
- `InMemoryEmbedder`
- `InMemoryVectorStore`
- `InMemoryLLM`
- `SQLiteVectorStore`
- `DefaultRequestContextHook`
- `NoOpAuthHook`
- `DefaultSafetyGuard`
- `InMemoryRateLimiter`
- `StructuredLogging`
- `JSONLogFormatter`
- `OpenTelemetryTracing`

`InMemoryEmbedder`, `InMemoryVectorStore`, and `InMemoryLLM` are development/test providers. They are deterministic and convenient for local testing, but they are not production embedding, vector-search, or generation backends.

Optional provider exports are available only when their extras are installed:

- `ChromaVectorStore` with `syrag[chroma]`
- `FAISSVectorStore` with `syrag[faiss]`
- `GoogleEmbedder` and `GoogleLLM` with `syrag[google]`
- `LangChainTextSplitterChunker` with `syrag[langchain]`
- `OpenAIEmbedder` and `OpenAILLM` with `syrag[openai]`

Testing helpers are available when `syrag[testing]` is installed.

See [Compatibility Matrix](./compatibility.md) for the full extras list, dependency boundaries, and common install combinations.

## Advanced Surface

SyRAG also exports registry, bootstrap, dependency-resolution, and default pipeline classes for advanced users building custom application factories or provider packages.

These APIs are public enough to use in `0.x`, but they are expected to evolve faster than schemas, protocols, and route decorators.

## Compatibility Rule

For `0.x` releases, SyRAG should avoid unnecessary top-level export churn. New exports should be intentional and covered by tests. Breaking changes to primary application models, protocols, and route decorators should be called out in the changelog.
