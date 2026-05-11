# LangChain Integration

LangChain is useful when you want chains, agents, tools, and runnable composition around a RAG service. SyRAG is useful when you want the RAG API boundary, provider contracts, request context, guardrails, and operations surface to stay explicit.

The most production-friendly pattern is to run SyRAG as a service and call it from LangChain.

The complete LangChain + Qdrant script is available at [`examples/integrations/langchain_qdrant_rag.py`](../../examples/integrations/langchain_qdrant_rag.py).
The complete LangChain agent script is available at [`examples/integrations/langchain_syrag_agent.py`](../../examples/integrations/langchain_syrag_agent.py).
The complete SyRAG route adapter script is available at [`examples/integrations/langchain_syrag_routes.py`](../../examples/integrations/langchain_syrag_routes.py).

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

## Plug LangChain Strategies Into SyRAG Routes

Use this when SyRAG should remain the FastAPI service boundary, but LangChain should provide mature strategy implementations such as text splitting or retrieval.

Install:

```bash
pip install "syrag[chroma,langchain,openai,server]" langchain-openai langchain-qdrant qdrant-client
```

The app in [`examples/integrations/langchain_syrag_routes.py`](../../examples/integrations/langchain_syrag_routes.py) wires a LangChain text splitter into `/ingest` and a LangChain Qdrant retriever into `/query`:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

from syrag.integrations.langchain import LangChainRetrieverStrategy, LangChainTextChunker


langchain_chunker = LangChainTextChunker(
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
)
langchain_retriever = langchain_vector_store.as_retriever(search_kwargs={"k": 5})


@syrag.ingest(
    "/ingest",
    chunker=langchain_chunker,
    embedder=syrag_embedder,
    vector_store=syrag_storage,
)
async def ingest(request: IngestRequest) -> IngestRequest:
    return request.model_copy(
        update={
            "collection": request.collection or "support",
            "metadata": {"source": "langchain-text-splitter", **request.metadata},
        }
    )


@syrag.query(
    "/query",
    embedder=syrag_embedder,
    vector_store=retriever_owned_vector_store,
    llm=syrag_llm,
    retrieval_strategy=LangChainRetrieverStrategy(retriever=langchain_retriever),
)
async def query(request: QueryRequest) -> QueryRequest:
    return request.model_copy(update={"top_k": min(request.top_k, 5)})
```

In the current route contract, SyRAG still requires an `embedder` and `vector_store` for query routes. When a `LangChainRetrieverStrategy` owns retrieval, the adapter ignores SyRAG’s `query_embedding` and `vector_store`; the example uses a small placeholder vector store to make that boundary explicit.

## Make SyRAG A LangChain Agent Tool

Install:

```bash
pip install "syrag[server]" langchain httpx
```

Run a SyRAG app, then expose its `/query` route as a tool the agent can decide to call:

```python
import httpx
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def ask_syrag(question: str, collection: str = "support", tenant_id: str | None = None) -> str:
    """Ask the SyRAG knowledge service for a grounded answer with citations."""
    with httpx.Client(base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/query",
            json={
                "query": question,
                "collection": collection,
                "tenant_id": tenant_id,
                "top_k": 5,
            },
            headers={"x-tenant-id": tenant_id} if tenant_id else None,
        )
        response.raise_for_status()
        payload = response.json()
        return f"{payload['answer']}\n\nCitations: {payload.get('citations', [])}"


agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[ask_syrag],
    system_prompt=(
        "You are a support agent. Use ask_syrag for questions about the private "
        "knowledge base. Do not answer private-knowledge questions from memory."
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the support collection for tenant-a. "
                    "What does SyRAG provide?"
                ),
            }
        ]
    }
)

print(result["messages"][-1].content)
```

This keeps SyRAG responsible for retrieval, generation, citations, request IDs, tenant handling, and error normalization. LangChain remains the agent orchestration layer.

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
