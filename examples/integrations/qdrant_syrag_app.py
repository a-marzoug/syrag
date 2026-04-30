from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from syrag import (
    DocumentChunk,
    Embedder,
    IngestRequest,
    InMemoryEmbedder,
    InMemoryLLM,
    QueryRequest,
    RetrievedChunk,
    Settings,
    SyRAG,
    VectorStore,
)
from syrag.protocols import EmbeddingVector


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


async def build_app() -> SyRAG:
    embedder: Embedder = InMemoryEmbedder(dimensions=16)
    vector_size = len((await embedder.embed(["dimension probe"]))[0])
    vector_store = QdrantVectorStore(
        client=QdrantClient(path=".syrag/qdrant"),
        collection_name="support_docs",
        vector_size=vector_size,
    )

    syrag = SyRAG(
        title="Support Bot",
        version="0.1.0",
        description="SyRAG backed by Qdrant",
        settings=Settings(),
    )
    syrag.register_embedder("default", embedder)
    syrag.register_vector_store("default", vector_store)
    syrag.register_llm("default", InMemoryLLM())
    syrag.configure_defaults(embedder="default", vector_store="default", llm="default")

    @syrag.ingest("/ingest")
    async def ingest(request: IngestRequest) -> IngestRequest:
        return request

    @syrag.query("/query")
    async def query(request: QueryRequest) -> QueryRequest:
        return request

    return syrag


syrag_app = asyncio.run(build_app())
app = syrag_app.api
