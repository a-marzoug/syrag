# Provider Examples

SyRAG keeps provider setup explicit. Register the components you want to use, then choose them as app defaults.

## Chroma Local Vector Store

Use Chroma when you want a specialized local vector database with persistent storage.

Install:

```bash
pip install "syrag[chroma]"
```

Configure:

```python
from pathlib import Path

from syrag import ChromaVectorStore, InMemoryEmbedder, InMemoryLLM, Settings, SyRAG

syrag = SyRAG(
    title="Support Bot",
    version="0.1.0",
    description="Chroma-backed SyRAG app",
    settings=Settings(),
)

syrag.register_embedder("default", InMemoryEmbedder())
syrag.register_vector_store(
    "default",
    ChromaVectorStore(path=Path(".syrag/chroma"), collection_name="support_docs"),
)
syrag.register_llm("default", InMemoryLLM())
syrag.configure_defaults(embedder="default", vector_store="default", llm="default")
```

## SQLite Vector Store

Use SQLite when you want a lightweight persistent store with no extra vector database dependency. It is useful for demos, small local projects, and tests that need data to survive process restarts.

```python
from pathlib import Path

from syrag import InMemoryEmbedder, InMemoryLLM, SQLiteVectorStore, Settings, SyRAG

syrag = SyRAG(
    title="Support Bot",
    version="0.1.0",
    description="SQLite-backed SyRAG app",
    settings=Settings(),
)

syrag.register_embedder("default", InMemoryEmbedder())
syrag.register_vector_store("default", SQLiteVectorStore(Path(".syrag/documents.sqlite3")))
syrag.register_llm("default", InMemoryLLM())
syrag.configure_defaults(embedder="default", vector_store="default", llm="default")
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

syrag.register_embedder(
    "default",
    OpenAIEmbedder(api_key=api_key, model="text-embedding-3-small"),
)
syrag.register_vector_store(
    "default",
    ChromaVectorStore(path=Path(".syrag/chroma"), collection_name="support_docs"),
)
syrag.register_llm("default", OpenAILLM(api_key=api_key, model="gpt-4.1-mini"))
syrag.configure_defaults(embedder="default", vector_store="default", llm="default")
```

## Route Shape

Provider choice does not change your route handlers:

```python
from syrag import IngestRequest, QueryRequest


@syrag.ingest("/ingest")
async def ingest(request: IngestRequest) -> IngestRequest:
    return request


@syrag.query("/query")
async def query(request: QueryRequest) -> QueryRequest:
    return request
```
