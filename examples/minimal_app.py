import os
from pathlib import Path

from syrag import (
    ChromaVectorStore,
    IngestRequest,
    OpenAIEmbedder,
    OpenAILLM,
    QueryRequest,
    Settings,
    SyRAG,
)

syrag = SyRAG(
    title="Support Bot",
    version="0.1.0",
    description="Minimal SyRAG app",
    settings=Settings(),
)

SUPPORT_COLLECTION = "support"

embedder = OpenAIEmbedder(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small",
)
vector_store = ChromaVectorStore(
    path=Path(".syrag/chroma"),
    collection_name="support_docs",
)
llm = OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4.1-mini")


@syrag.ingest("/ingest", embedder=embedder, vector_store=vector_store)
async def ingest(request: IngestRequest) -> IngestRequest:
    return request.model_copy(
        update={
            "collection": request.collection or SUPPORT_COLLECTION,
            "metadata": {"source": "api", **request.metadata},
        }
    )


@syrag.query("/query", embedder=embedder, vector_store=vector_store, llm=llm)
async def query(request: QueryRequest) -> QueryRequest:
    return request.model_copy(
        update={
            "collection": request.collection or SUPPORT_COLLECTION,
            "top_k": min(request.top_k, 5),
        }
    )


api = syrag.api
