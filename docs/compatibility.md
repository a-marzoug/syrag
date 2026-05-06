# Compatibility Matrix

SyRAG keeps the core package small. Optional integrations live behind extras so applications only install the provider SDKs they use.

Supported Python versions for the current release line are `3.12` and `3.13`.

## Core Package

Install:

```bash
pip install syrag
```

| Surface | Included | Notes |
| --- | --- | --- |
| Framework | `SyRAG`, `create_app`, typed schemas, decorators, registry, bootstrap helpers | Stable public surface for `0.x` usage. |
| Providers | `SQLiteVectorStore`, `PassThroughChunker` | SQLite is lightweight persistence, not a specialized production vector database. |
| Dev/test providers | `InMemoryEmbedder`, `InMemoryVectorStore`, `InMemoryLLM` | Intended for framework development and tests only. |
| Operations | structured errors, structured logging, OpenTelemetry API integration, request context, guardrails | Exporter and backend configuration stays in the application. |

## Optional Extras

| Extra | Install | Dependencies | Public exports | Intended use |
| --- | --- | --- | --- | --- |
| `chroma` | `pip install "syrag[chroma]"` | `chromadb>=1.0.0` | `ChromaVectorStore` | Local persistent vector database or Chroma-backed deployments. |
| `faiss` | `pip install "syrag[faiss]"` | `faiss-cpu>=1.8.0` | `FAISSVectorStore` | Local vector indexing without running a vector database service. Metadata is stored in process. |
| `google` | `pip install "syrag[google]"` | `google-genai>=1.0.0` | `GoogleEmbedder`, `GoogleLLM` | Gemini embeddings and generation through the Google Gen AI SDK. |
| `langchain` | `pip install "syrag[langchain]"` | `langchain-core>=1.0.0`, `langchain-text-splitters>=0.3.0` | `LangChainTextChunker`, `LangChainRetrieverStrategy` | Use LangChain text splitters and retrievers behind SyRAG protocols. |
| `openai` | `pip install "syrag[openai]"` | `httpx>=0.28.1` | `OpenAIEmbedder`, `OpenAILLM` | OpenAI embeddings and generation through direct HTTP adapters. |
| `server` | `pip install "syrag[server]"` | `uvicorn[standard]>=0.44.0` | CLI/server runtime dependency | Local ASGI serving with the bundled `syrag` command or `uvicorn`. |
| `testing` | `pip install "syrag[testing]"` | `httpx>=0.28.1` | `create_test_app`, `create_test_client`, `seed_documents`, fake providers | Downstream application tests without external model or vector-store services. |
| `all` | `pip install "syrag[all]"` | `chromadb`, `faiss-cpu`, `google-genai`, `httpx`, `langchain-core`, `langchain-text-splitters`, `uvicorn[standard]` | Runtime integrations from `chroma`, `faiss`, `google`, `langchain`, `openai`, and `server` | Convenience install for local development with all runtime integrations. |

`all` intentionally does not include `testing`. Install `syrag[testing]` separately when downstream test helpers are needed.

## Common Combinations

| Use case | Install |
| --- | --- |
| OpenAI plus Chroma | `pip install "syrag[openai,chroma,server]"` |
| OpenAI plus FAISS | `pip install "syrag[openai,faiss,server]"` |
| Google plus Chroma | `pip install "syrag[google,chroma,server]"` |
| Google plus FAISS | `pip install "syrag[google,faiss,server]"` |
| LangChain splitting plus OpenAI and Chroma | `pip install "syrag[langchain,openai,chroma,server]"` |
| Application tests | `pip install "syrag[testing]"` |

## Compatibility Rules

- Core imports must not require optional provider SDKs.
- Optional provider exports are present only when their extras are installed.
- Provider adapters should keep SDK-specific behavior inside `syrag.providers.*`.
- Framework strategy adapters should live behind optional integration extras.
- Prefer adapters for LangChain, LlamaIndex, and similar ecosystems before adding more first-party strategy implementations.
- New extras should be documented here and covered by packaging tests.
