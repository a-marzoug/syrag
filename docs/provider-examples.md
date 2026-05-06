# Provider Examples

SyRAG keeps provider setup explicit. Register the components you want to use, then choose them as app defaults.

## Chroma Local Vector Store

Use Chroma when you want a specialized local vector database with persistent storage.

Install:

```bash
pip install "syrag[chroma,openai]"
```

Configure:

```python
import os
from pathlib import Path

from syrag import ChromaVectorStore, OpenAIEmbedder, OpenAILLM, Settings, SyRAG

api_key = os.environ["OPENAI_API_KEY"]

syrag = SyRAG(
    title="Support Bot",
    version="0.1.0",
    description="Chroma-backed SyRAG app",
    settings=Settings(),
)

embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-3-small")
vector_store = ChromaVectorStore(
    path=Path(".syrag/chroma"),
    collection_name="support_docs",
)
llm = OpenAILLM(api_key=api_key, model="gpt-4.1-mini")
```

## FAISS Local Vector Store

Use FAISS when you want a specialized local vector index without running a vector database service. SyRAG's FAISS adapter keeps metadata in process and rebuilds the local index on upserts, so use it for local apps, prototypes, and small deployments before moving to a durable remote vector database.

Install:

```bash
pip install "syrag[faiss,openai]"
```

Configure:

```python
import os

from syrag import FAISSVectorStore, OpenAIEmbedder, OpenAILLM, Settings, SyRAG

api_key = os.environ["OPENAI_API_KEY"]

syrag = SyRAG(
    title="Support Bot",
    version="0.1.0",
    description="FAISS-backed SyRAG app",
    settings=Settings(),
)

embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-3-small")
vector_store = FAISSVectorStore(dimensions=1536)
llm = OpenAILLM(api_key=api_key, model="gpt-4.1-mini")
```

## SQLite Vector Store

Use SQLite when you want a lightweight persistent store with no extra vector database dependency. It is useful for demos, small local projects, and tests that need data to survive process restarts. For production vector search, prefer Chroma, Qdrant, or another vector database.

```python
import os
from pathlib import Path

from syrag import OpenAIEmbedder, OpenAILLM, SQLiteVectorStore, Settings, SyRAG

api_key = os.environ["OPENAI_API_KEY"]

syrag = SyRAG(
    title="Support Bot",
    version="0.1.0",
    description="SQLite-backed SyRAG app",
    settings=Settings(),
)

embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-3-small")
vector_store = SQLiteVectorStore(Path(".syrag/documents.sqlite3"))
llm = OpenAILLM(api_key=api_key, model="gpt-4.1-mini")
```

## OpenAI Embedder And LLM

Use OpenAI when you want hosted embeddings and generation. Keep the API key in your environment rather than hard-coding it.

Install:

```bash
pip install "syrag[openai]"
```

Configure:

```python
import os
from pathlib import Path

from syrag import ChromaVectorStore, OpenAIEmbedder, OpenAILLM, Settings, SyRAG

api_key = os.environ["OPENAI_API_KEY"]

syrag = SyRAG(
    title="Support Bot",
    version="0.1.0",
    description="OpenAI-backed SyRAG app",
    settings=Settings(),
)

embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-3-small")
vector_store = ChromaVectorStore(
    path=Path(".syrag/chroma"),
    collection_name="support_docs",
)
llm = OpenAILLM(api_key=api_key, model="gpt-4.1-mini")
```

## Google Embedder And LLM

Use Google when you want Gemini embeddings and generation through the Google Gen AI SDK. Use `api_key` for the Gemini Developer API, or configure `vertexai=True` with a Google Cloud project and location for Vertex AI.

Install:

```bash
pip install "syrag[google,chroma]"
```

Configure:

```python
import os
from pathlib import Path

from syrag import ChromaVectorStore, GoogleEmbedder, GoogleLLM, Settings, SyRAG

api_key = os.environ["GOOGLE_API_KEY"]

syrag = SyRAG(
    title="Support Bot",
    version="0.1.0",
    description="Google-backed SyRAG app",
    settings=Settings(),
)

embedder = GoogleEmbedder(api_key=api_key, model="gemini-embedding-001")
vector_store = ChromaVectorStore(
    path=Path(".syrag/chroma"),
    collection_name="support_docs",
)
llm = GoogleLLM(api_key=api_key, model="gemini-2.5-flash")
```

## In-Memory Providers

Use in-memory providers for tests, examples that should not call external services, and framework development.

They are not production providers:

- `InMemoryEmbedder` is deterministic and hash-based, not semantic.
- `InMemoryVectorStore` stores data only in the current Python process.
- `InMemoryLLM` produces simple grounded test responses, not model-quality answers.

```python
from syrag import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore

embedder = InMemoryEmbedder()
vector_store = InMemoryVectorStore()
llm = InMemoryLLM()
```

## LangChain Text Splitter

Use `LangChainTextChunker` when you want SyRAG ingest routes to reuse a LangChain text splitter instead of a SyRAG-specific chunking implementation.

Install:

```bash
pip install "syrag[langchain,openai,chroma]"
```

Configure:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from syrag import IngestRequest, LangChainTextChunker

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1_000,
    chunk_overlap=200,
)
chunker = LangChainTextChunker(text_splitter=text_splitter)


@syrag.ingest("/ingest", chunker=chunker, embedder=embedder, vector_store=vector_store)
async def ingest(request: IngestRequest) -> IngestRequest:
    return request.model_copy(update={"collection": request.collection or "support"})
```

## LangChain Retriever Strategy

Use `LangChainRetrieverStrategy` when a LangChain retriever should own query-time retrieval while SyRAG still owns the API route, request validation, prompt assembly, generation policy, and LLM response.

```python
from syrag import LangChainRetrieverStrategy, QueryRequest

retrieval_strategy = LangChainRetrieverStrategy(retriever=retriever)


@syrag.query(
    "/query",
    embedder=embedder,
    vector_store=vector_store,
    retrieval_strategy=retrieval_strategy,
    llm=llm,
)
async def query(request: QueryRequest) -> QueryRequest:
    return request.model_copy(update={"collection": request.collection or "support"})
```

The adapter calls `retriever.ainvoke(query)` when available, otherwise it runs `retriever.invoke(query)` off the event loop. Configure LangChain-specific search parameters, filters, and `k` on the retriever itself before passing it to SyRAG.

## Route Shape

Provider choice does not change your route handlers:

```python
from syrag import IngestRequest, QueryRequest


@syrag.ingest("/ingest")
async def ingest(request: IngestRequest) -> IngestRequest:
    return request.model_copy(update={"collection": request.collection or "support"})


@syrag.query("/query")
async def query(request: QueryRequest) -> QueryRequest:
    return request.model_copy(update={"collection": request.collection or "support"})
```
