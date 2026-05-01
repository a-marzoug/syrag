from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from syrag import (
    LLM,
    DocumentChunk,
    Embedder,
    IngestRequest,
    OpenAIEmbedder,
    OpenAILLM,
    QueryRequest,
    RetrievedChunk,
    Settings,
    SyRAG,
    VectorStore,
)
from syrag.protocols import EmbeddingVector

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSIONS = 1536
OPENAI_LLM_MODEL = "gpt-4.1-mini"
QDRANT_COLLECTION = "support_docs"
SUPPORT_COLLECTION = "support"


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        *,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    async def upsert(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
        collection: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{collection}:{tenant_id}:{chunk.chunk_id}")),
                vector=[float(value) for value in embedding],
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "content": chunk.content,
                    "collection": collection or "",
                    "tenant_id": tenant_id or "",
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "metadata": dict(chunk.metadata),
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    async def query(
        self,
        *,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=[float(value) for value in query_embedding],
            query_filter=self._filter_for(collection, tenant_id, filters),
            limit=top_k,
            with_payload=True,
        )
        chunks: list[RetrievedChunk] = []
        for point in result.points:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(payload["chunk_id"]),
                    source_id=str(payload["source_id"]),
                    content=str(payload["content"]),
                    score=float(point.score or 0.0),
                    metadata=metadata if isinstance(metadata, dict) else {},
                    page_number=payload.get("page_number"),
                    chunk_index=int(payload.get("chunk_index", 0)),
                )
            )
        return chunks

    def _filter_for(
        self,
        collection: str | None,
        tenant_id: str | None,
        filters: Mapping[str, Any] | None,
    ) -> models.Filter:
        must = [
            models.FieldCondition(
                key="collection",
                match=models.MatchValue(value=collection or ""),
            ),
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id or ""),
            ),
        ]
        for key, value in (filters or {}).items():
            if isinstance(value, str | int | float | bool):
                must.append(
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=value),
                    )
                )
        return models.Filter(must=must)


def build_embedder() -> Embedder:
    return OpenAIEmbedder(
        api_key=os.environ["OPENAI_API_KEY"],
        model=OPENAI_EMBEDDING_MODEL,
    )


def build_llm() -> LLM:
    return OpenAILLM(
        api_key=os.environ["OPENAI_API_KEY"],
        model=OPENAI_LLM_MODEL,
    )


def build_vector_store() -> VectorStore:
    return QdrantVectorStore(
        client=QdrantClient(path=".syrag/qdrant"),
        collection_name=QDRANT_COLLECTION,
        vector_size=OPENAI_EMBEDDING_DIMENSIONS,
    )


async def build_app() -> SyRAG:
    embedder = build_embedder()
    vector_store = build_vector_store()
    llm = build_llm()

    syrag = SyRAG(
        title="Support Bot",
        version="0.1.0",
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
