# LangChain Integration

LangChain is useful when you want chains, agents, tools, and runnable composition around a RAG service. SyRAG is useful when you want the RAG API boundary, provider contracts, request context, guardrails, and operations surface to stay explicit.

The most production-friendly pattern is to run SyRAG as a service and call it from LangChain.

The complete LangChain + Qdrant script is available at [`examples/integrations/langchain_qdrant_rag.py`](../../examples/integrations/langchain_qdrant_rag.py).

## Full LangChain + Qdrant RAG Pipeline

Use this when LangChain owns the whole local RAG pipeline and Qdrant is the vector store.

Install:

```bash
pip install langchain langchain-openai langchain-qdrant qdrant-client
```

Create `langchain_qdrant_rag.py`:

```python
import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


COLLECTION = "support_docs"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)
client = QdrantClient(path=".syrag/langchain-qdrant")
vector_size = len(embeddings.embed_query("dimension probe"))

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION,
    embedding=embeddings,
)

documents = [
    Document(
        page_content="SyRAG exposes typed ingest and query routes for RAG services.",
        metadata={"source": "overview", "topic": "api"},
    ),
    Document(
        page_content="Qdrant stores vectors and payloads for similarity search.",
        metadata={"source": "qdrant", "topic": "vector-store"},
    ),
]
vector_store.add_documents(documents)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer from the provided context. If the context is insufficient, say you do not know.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)
model = ChatOpenAI(model="gpt-4.1-mini", api_key=os.environ["OPENAI_API_KEY"])


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("What does SyRAG expose?")
print(answer)
```

This pipeline is pure LangChain. Use it when you do not need SyRAG’s service boundary, request context, route validation, or framework-level error model.

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
