# LlamaIndex Integration

LlamaIndex is strong for data connectors, indexing workflows, and query engines. SyRAG is strong as a typed service layer for RAG APIs, provider contracts, tenant-aware request handling, and operations.

The cleanest production pattern is to let LlamaIndex prepare or retrieve data when needed, while SyRAG owns the API boundary.

## Send LlamaIndex Documents To SyRAG

Install:

```bash
pip install "syrag[server]" llama-index httpx
```

Load documents with LlamaIndex and ingest them into a running SyRAG service:

```python
import httpx
from llama_index.core import SimpleDirectoryReader


async def ingest_directory(
    *,
    directory: str,
    collection: str,
    tenant_id: str | None = None,
) -> None:
    documents = SimpleDirectoryReader(directory).load_data()
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        for index, document in enumerate(documents):
            response = await client.post(
                "/ingest",
                json={
                    "documents": [document.text],
                    "collection": collection,
                    "tenant_id": tenant_id,
                    "metadata": {
                        "source_id": document.metadata.get("file_name", f"document-{index}"),
                        "llamaindex_document_id": document.id_,
                    },
                },
            )
            response.raise_for_status()
```

This is a good bridge when you like LlamaIndex loaders but want SyRAG to handle serving, observability, and request-level behavior.

## Wrap A SyRAG Embedder For LlamaIndex

Use this when you want LlamaIndex indexes to use the same embedding implementation as your SyRAG app.

```python
import asyncio
from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from syrag import Embedder, InMemoryEmbedder


class SyRAGLlamaIndexEmbedding(BaseEmbedding):
    _embedder: Embedder = PrivateAttr()

    def __init__(self, embedder: Embedder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._embedder = embedder

    @classmethod
    def class_name(cls) -> str:
        return "syrag"

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return _run_async(self._embedder.embed([text]))[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return _run_async(self._embedder.embed(texts))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return (await self._embedder.embed([query]))[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self._embedder.embed([text]))[0]


def _run_async(awaitable: object) -> object:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)  # type: ignore[arg-type]
    msg = "Use LlamaIndex async APIs inside an active event loop."
    raise RuntimeError(msg)


embed_model = SyRAGLlamaIndexEmbedding(InMemoryEmbedder())
```

Use it with a LlamaIndex index:

```python
from llama_index.core import Document, VectorStoreIndex

documents = [
    Document(text="SyRAG exposes typed ingest and query routes."),
    Document(text="SyRAG supports tenant-aware retrieval."),
]

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = index.as_query_engine()
response = query_engine.query("What does SyRAG expose?")

print(response)
```

## Call SyRAG From A LlamaIndex Tool

If an agent or workflow should call your SyRAG service, expose `/query` as a tool:

```python
from typing import Any

import httpx
from llama_index.core.tools import FunctionTool


async def ask_syrag(question: str, collection: str = "support") -> str:
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        response = await client.post(
            "/query",
            json={"query": question, "collection": collection, "top_k": 5},
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return str(payload["answer"])


syrag_tool = FunctionTool.from_defaults(
    async_fn=ask_syrag,
    name="ask_syrag",
    description="Ask the SyRAG knowledge service for grounded answers.",
)
```

## When To Use This Pattern

Use LlamaIndex with SyRAG when:

- You want LlamaIndex readers and ingestion utilities.
- You want LlamaIndex query engines for experiments, but SyRAG for serving.
- You are gradually moving from notebooks or indexes toward a production API boundary.
