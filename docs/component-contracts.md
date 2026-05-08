# Component Contracts

## Why Contracts Matter

SyRAG is built around runtime-checkable protocols. The goal is to let applications replace providers and policy hooks without rewriting route shape or losing observability and error guarantees.

Contracts should be:

- typed
- async-first where IO is involved
- validated at runtime
- explicit about request scope and tenancy where relevant

## Core Rules

Every public extension point follows these rules:

- inputs and outputs are defined with Pydantic models
- network-bound operations are asynchronous
- storage and retrieval operations accept collection and tenant context
- errors are normalized into SyRAG stage-aware exceptions

## Core Roles

### Chunker

Transforms `SourceDocument` values into `DocumentChunk` values during ingest.

Responsibilities:

- preserve source identity and metadata
- emit stable chunk IDs
- produce chunk indexes and optional page numbers

The first-party chunker currently includes `PassThroughChunker` for one chunk per source document. More specialized chunking should be provided through framework adapters where possible.

### Embedder

Converts text into vectors for ingest and query retrieval.

Responsibilities:

- return one vector per input string
- stay neutral to local and hosted implementations
- keep provider-specific transport concerns inside the adapter

### Vector Store

Persists and queries embeddings with metadata filters.

Responsibilities:

- upsert vectors and chunk metadata
- query by embedding with filters
- isolate data by collection or tenant
- return `RetrievedChunk` results with scores and source metadata

The vector store boundary is where namespace isolation becomes non-negotiable.

### Retrieval Strategy

Owns query-time retrieval behavior rather than raw storage access. The default strategy currently:

- embeds the query
- queries the vector store
- returns top-k retrieved chunks

This seam exists so hybrid retrieval, reranking, or query rewriting can be added later without changing route handlers.

### Prompt Assembler

Builds the grounded prompt package from the query and retrieved chunks.

Responsibilities:

- receive the `QueryRequest` and retrieved context
- produce `AssembledPrompt`
- stay independent from provider-specific generation APIs

### Reranker

Reorders or filters retrieved chunks before prompt assembly.

Responsibilities:

- receive the original `QueryRequest`
- receive candidate `RetrievedChunk` values
- return the context that should continue into prompt assembly
- keep reranking provider details outside route handlers

Use `RerankingRetrievalStrategy` to compose a reranker with any retrieval strategy that already implements the SyRAG `RetrievalStrategy` protocol.

### Generation Policy

Applies generation-time policy to an assembled prompt before the LLM sees it.

Responsibilities:

- enforce context limits or prompt shaping
- inject system instructions
- decide whether citations are required
- produce the final `GenerationRequest`

### LLM

Generates the final grounded answer from a `GenerationRequest`.

Responsibilities:

- consume the final prompt plus retrieved context
- return `RAGResponse`
- populate citations and usage metadata when available

## Shared Data Expectations

The shared schema family currently includes:

- document and chunk metadata
- request context
- assembled prompts and generation requests
- scored retrieval results
- final grounded responses with citations and usage
- structured error envelopes

At the framework level, the default path expects grounded responses with citations. A custom generation policy can relax that requirement for a specific route.

## Request-Scope Hooks

SyRAG also exposes protocol seams outside the main retrieval/generation path:

- `RequestContextHook`
- `AuthHook`
- `RateLimiter`
- `SafetyGuard`

These hooks let applications enforce request identity, auth, throttling, and payload safety without moving that logic into route handlers.

## Registration Model

Applications register concrete providers by name on the app registry, then choose defaults separately:

```python
from syrag import SyRAG, InMemoryEmbedder, Settings

app = SyRAG(
    title="Support Bot",
    version="0.1.0",
    description="Internal support assistant",
    settings=Settings(),
)

app.register_embedder("default", InMemoryEmbedder())
app.configure_defaults(embedder="default")
```

This example uses `InMemoryEmbedder` only to demonstrate the registry mechanics. In-memory providers are intended for development and tests, not production retrieval quality.

The current registry supports:

- first-party built-ins
- manual application registration
- app-level default selection

It does not yet implement plugin discovery.

## Compatibility Promise

SyRAG keeps a narrow compatibility promise:

- application code depends on SyRAG protocols and schemas
- adapters depend on external provider SDKs
- switching providers should not require route rewrites
- operational hooks should compose without changing provider implementations

That boundary is what keeps the framework small while still leaving room for more specialized providers later.
