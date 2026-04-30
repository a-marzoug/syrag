# Cookbook

The cookbook shows how SyRAG fits into real application stacks. These recipes are intentionally integration-focused: they show the seams between SyRAG and neighboring tools rather than only repeating the core quick start.

## Recipes

- [Qdrant vector store pipeline](./qdrant.md)
- [LangChain integration](./langchain.md)
- [LangGraph RAG pipeline](./langgraph.md)
- [LlamaIndex integration](./llamaindex.md)

## Integration Patterns

Use SyRAG in one of three ways:

- As the service boundary: expose `/ingest` and `/query`, then call SyRAG from another framework over HTTP.
- As the provider layer: reuse SyRAG embedders, vector stores, or LLM adapters inside another framework.
- As the application shell: keep SyRAG routes and lifecycle, but register providers that wrap another framework.

The service-boundary approach is usually the cleanest production shape. It keeps request context, tenant scoping, rate limiting, guardrails, observability, and errors inside SyRAG while still allowing LangChain, LlamaIndex, or agent frameworks to call it.

Full scripts are also available in [`examples/integrations`](../../examples/integrations).
