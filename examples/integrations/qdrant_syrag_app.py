from __future__ import annotations

import asyncio
import os
from pathlib import Path

from syrag import (
    IngestRequest,
    OpenAIEmbedder,
    OpenAILLM,
    QdrantVectorStore,
    QueryRequest,
    Settings,
    SyRAG,
)

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSIONS = 1536
OPENAI_LLM_MODEL = "gpt-4.1-mini"
QDRANT_COLLECTION = "support_docs"
SUPPORT_COLLECTION = "support"


def build_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder(
        api_key=os.environ["OPENAI_API_KEY"],
        model=OPENAI_EMBEDDING_MODEL,
    )


def build_llm() -> OpenAILLM:
    return OpenAILLM(
        api_key=os.environ["OPENAI_API_KEY"],
        model=OPENAI_LLM_MODEL,
    )


def build_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        path=Path(".syrag/qdrant"),
        collection_name=QDRANT_COLLECTION,
        dimensions=OPENAI_EMBEDDING_DIMENSIONS,
    )


async def build_app() -> SyRAG:
    embedder = build_embedder()
    vector_store = build_vector_store()
    llm = build_llm()

    syrag = SyRAG(
        title="Support Bot",
        version="0.3.0",
        description="SyRAG backed by Qdrant",
        settings=Settings(),
    )

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

    return syrag


syrag_app = asyncio.run(build_app())
app = syrag_app.api
