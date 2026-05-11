# LlamaIndex Integration

LlamaIndex is strong for data connectors, indexing workflows, and query engines. SyRAG is strong as a typed service layer for RAG APIs, provider contracts, tenant-aware request handling, and operations.

The cleanest production pattern is to let LlamaIndex prepare or retrieve data when needed, while SyRAG owns the API boundary.

The complete LlamaIndex + Qdrant script is available at [`examples/integrations/llamaindex_qdrant_rag.py`](../../examples/integrations/llamaindex_qdrant_rag.py).
The complete SyRAG route adapter script is available at [`examples/integrations/llamaindex_syrag_routes.py`](../../examples/integrations/llamaindex_syrag_routes.py).

## Full LlamaIndex + Qdrant RAG Pipeline

Use this when LlamaIndex owns ingestion, indexing, and query execution while Qdrant stores vectors.

Install:

```bash
pip install llama-index llama-index-vector-stores-qdrant llama-index-embeddings-openai qdrant-client
```

Create `llamaindex_qdrant_rag.py`:

```python
import os

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)
Settings.llm = OpenAI(
    model="gpt-4.1-mini",
    api_key=os.environ["OPENAI_API_KEY"],
)

client = QdrantClient(path=".syrag/llamaindex-qdrant")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="support_docs",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = [
    Document(
        text="SyRAG exposes typed ingest and query routes for RAG services.",
        metadata={"source": "overview", "topic": "api"},
    ),
    Document(
        text="Qdrant stores vectors and payloads for semantic retrieval.",
        metadata={"source": "qdrant", "topic": "vector-store"},
    ),
]

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query("What does SyRAG expose?")
print(response)
```

This pipeline is pure LlamaIndex. Use it for experiments, data-heavy indexing workflows, or when LlamaIndex query engines are the application boundary.

## Plug LlamaIndex Strategies Into SyRAG Routes

Use this when SyRAG should own the HTTP API, request context, guardrails, and response model, while LlamaIndex provides node parsing or retrieval.

Install:

```bash
pip install "syrag[chroma,llamaindex,openai,server]" llama-index-embeddings-openai llama-index-vector-stores-qdrant qdrant-client
```

The app in [`examples/integrations/llamaindex_syrag_routes.py`](../../examples/integrations/llamaindex_syrag_routes.py) wires a LlamaIndex node parser into `/ingest` and a LlamaIndex Qdrant-backed retriever into `/query`:

```python
from llama_index.core.node_parser import SentenceSplitter

from syrag.integrations.llamaindex import LlamaIndexNodeChunker, LlamaIndexRetrieverStrategy


llamaindex_chunker = LlamaIndexNodeChunker(
    node_parser=SentenceSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
)
llamaindex_retriever = llamaindex_index.as_retriever(similarity_top_k=5)


@syrag.ingest(
    "/ingest",
    chunker=llamaindex_chunker,
    embedder=syrag_embedder,
    vector_store=syrag_storage,
)
async def ingest(request: IngestRequest) -> IngestRequest:
    return request.model_copy(
        update={
            "collection": request.collection or "support",
            "metadata": {"source": "llamaindex-node-parser", **request.metadata},
        }
    )


@syrag.query(
    "/query",
    embedder=syrag_embedder,
    vector_store=retriever_owned_vector_store,
    llm=syrag_llm,
    retrieval_strategy=LlamaIndexRetrieverStrategy(retriever=llamaindex_retriever),
)
async def query(request: QueryRequest) -> QueryRequest:
    return request.model_copy(update={"top_k": min(request.top_k, 5)})
```

In the current route contract, SyRAG still requires an `embedder` and `vector_store` for query routes. When a `LlamaIndexRetrieverStrategy` owns retrieval, the adapter ignores SyRAG’s `query_embedding` and `vector_store`; the example uses a small placeholder vector store to make that boundary explicit.

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

## Wrap A SyRAG OpenAI Embedder For LlamaIndex

Use this when you want LlamaIndex indexes to use the same embedding implementation as your SyRAG app.

```python
import asyncio
import os
from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from syrag import Embedder, OpenAIEmbedder


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


embed_model = SyRAGLlamaIndexEmbedding(
    OpenAIEmbedder(
        api_key=os.environ["OPENAI_API_KEY"],
        model="text-embedding-3-small",
    )
)
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
