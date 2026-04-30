# LangChain Integration

LangChain is useful when you want chains, agents, tools, and runnable composition around a RAG service. SyRAG is useful when you want the RAG API boundary, provider contracts, request context, guardrails, and operations surface to stay explicit.

The most production-friendly pattern is to run SyRAG as a service and call it from LangChain.

## Call A SyRAG Query Route From A LangChain Runnable

Install:

```bash
pip install "syrag[server]" langchain-core httpx
```

Run a SyRAG app, then call its `/query` route from a LangChain runnable:

```python
from typing import Any

import httpx
from langchain_core.runnables import RunnableLambda


async def query_syrag(payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        response = await client.post(
            "/query",
            json={
                "query": payload["question"],
                "collection": payload.get("collection", "support"),
                "tenant_id": payload.get("tenant_id"),
                "top_k": payload.get("top_k", 5),
                "filters": payload.get("filters", {}),
            },
            headers={
                "x-tenant-id": payload["tenant_id"],
            }
            if payload.get("tenant_id")
            else None,
        )
        response.raise_for_status()
        return response.json()


syrag_query = RunnableLambda(query_syrag)

result = await syrag_query.ainvoke(
    {
        "question": "What does our refund policy say?",
        "collection": "support",
        "tenant_id": "tenant-a",
        "top_k": 3,
    }
)

print(result["answer"])
```

This keeps SyRAG responsible for retrieval, generation, citations, request IDs, tenant handling, and error normalization. LangChain remains the orchestration layer around that service call.

## Wrap A SyRAG Embedder For LangChain

Use this when you want a SyRAG embedder implementation to power a LangChain vector store.

```python
import asyncio
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings
from syrag import Embedder, InMemoryEmbedder


class SyRAGLangChainEmbeddings(Embeddings):
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _run_async(self.embedder.embed(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embedder.embed(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]


def _run_async(awaitable: object) -> object:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)  # type: ignore[arg-type]
    msg = "Use aembed_documents() or aembed_query() inside an active event loop."
    raise RuntimeError(msg)


embeddings = SyRAGLangChainEmbeddings(InMemoryEmbedder())
```

Then pass `embeddings` to any LangChain vector store that accepts an `Embeddings` implementation.

## Convert LangChain Documents For SyRAG Ingest

LangChain loaders usually produce `Document` values with `page_content` and `metadata`. SyRAG ingest accepts text documents plus shared metadata.

```python
from langchain_core.documents import Document
from syrag import IngestRequest


def ingest_request_from_documents(
    documents: list[Document],
    *,
    collection: str,
    tenant_id: str | None = None,
) -> IngestRequest:
    return IngestRequest(
        documents=[document.page_content for document in documents],
        collection=collection,
        tenant_id=tenant_id,
        metadata={
            "source_id": "langchain-import",
            "langchain_documents": len(documents),
        },
    )
```

For per-document metadata preservation, prefer calling the SyRAG HTTP ingest route in smaller batches grouped by source metadata.

## When To Use This Pattern

Use LangChain around SyRAG when:

- You already have LangChain agents or runnables.
- SyRAG should own the RAG service API and operational behavior.
- You want citations and framework-normalized errors returned to a larger chain.

Use direct SyRAG routes without LangChain when the application only needs ingest, query, and provider customization.
