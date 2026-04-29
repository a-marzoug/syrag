from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from syrag.protocols import EmbeddingVector, Filters, VectorStore
from syrag.schemas import DocumentChunk, RetrievedChunk


@dataclass(slots=True)
class SQLiteStoredDocument:
    chunk_id: str
    source_id: str
    content: str
    embedding: tuple[float, ...]
    metadata: dict[str, Any]
    page_number: int | None
    chunk_index: int


class SQLiteVectorStore(VectorStore):
    """Persistent SQLite-backed vector store for local and small-scale deployments."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    async def upsert(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
        collection: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self._upsert_sync(
            chunks,
            embeddings,
            collection,
            tenant_id,
        )

    async def query(
        self,
        *,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: Filters | None = None,
    ) -> list[RetrievedChunk]:
        return self._query_sync(
            query_embedding,
            top_k,
            collection,
            tenant_id,
            filters,
        )

    def _initialize_database(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_chunks (
                    collection_key TEXT NOT NULL,
                    tenant_key TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    page_number INTEGER,
                    chunk_index INTEGER NOT NULL,
                    PRIMARY KEY (collection_key, tenant_key, chunk_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS vector_chunks_namespace_idx
                ON vector_chunks(collection_key, tenant_key)
                """
            )

    def _upsert_sync(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
        collection: str | None,
        tenant_id: str | None,
    ) -> None:
        if len(chunks) != len(embeddings):
            msg = "chunks and embeddings must have the same length"
            raise ValueError(msg)

        collection_key = self._namespace_key(collection)
        tenant_key = self._namespace_key(tenant_id)
        rows = [
            (
                collection_key,
                tenant_key,
                chunk.chunk_id,
                chunk.source_id,
                chunk.content,
                json.dumps([float(value) for value in embedding]),
                json.dumps(chunk.metadata),
                chunk.page_number,
                chunk.chunk_index,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        with sqlite3.connect(self.database_path) as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO vector_chunks (
                    collection_key,
                    tenant_key,
                    chunk_id,
                    source_id,
                    content,
                    embedding_json,
                    metadata_json,
                    page_number,
                    chunk_index
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def _query_sync(
        self,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str | None,
        tenant_id: str | None,
        filters: Filters | None,
    ) -> list[RetrievedChunk]:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    source_id,
                    content,
                    embedding_json,
                    metadata_json,
                    page_number,
                    chunk_index
                FROM vector_chunks
                WHERE collection_key = ? AND tenant_key = ?
                """,
                (
                    self._namespace_key(collection),
                    self._namespace_key(tenant_id),
                ),
            ).fetchall()

        normalized_query = tuple(float(value) for value in query_embedding)
        documents = [
            self._deserialize_row(row)
            for row in rows
        ]
        filtered_documents = [
            document
            for document in documents
            if self._matches_filters(document.metadata, filters)
        ]
        ranked_documents = sorted(
            filtered_documents,
            key=lambda document: self._cosine_similarity(normalized_query, document.embedding),
            reverse=True,
        )

        return [
            RetrievedChunk(
                chunk_id=document.chunk_id,
                source_id=document.source_id,
                content=document.content,
                score=self._cosine_similarity(normalized_query, document.embedding),
                metadata=document.metadata,
                page_number=document.page_number,
                chunk_index=document.chunk_index,
            )
            for document in ranked_documents[:top_k]
        ]

    def _deserialize_row(self, row: sqlite3.Row) -> SQLiteStoredDocument:
        return SQLiteStoredDocument(
            chunk_id=str(row["chunk_id"]),
            source_id=str(row["source_id"]),
            content=str(row["content"]),
            embedding=tuple(float(value) for value in json.loads(str(row["embedding_json"]))),
            metadata=dict(json.loads(str(row["metadata_json"]))),
            page_number=int(row["page_number"]) if row["page_number"] is not None else None,
            chunk_index=int(row["chunk_index"]),
        )

    def _matches_filters(
        self,
        metadata: Mapping[str, Any],
        filters: Filters | None,
    ) -> bool:
        if not filters:
            return True
        return all(metadata.get(key) == value for key, value in filters.items())

    def _namespace_key(self, value: str | None) -> str:
        return value or ""

    def _cosine_similarity(
        self,
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        if len(left) != len(right):
            return 0.0

        numerator = sum(
            left_value * right_value for left_value, right_value in zip(left, right, strict=True)
        )
        left_magnitude = math.sqrt(sum(value * value for value in left))
        right_magnitude = math.sqrt(sum(value * value for value in right))
        if left_magnitude == 0.0 or right_magnitude == 0.0:
            return 0.0
        return numerator / (left_magnitude * right_magnitude)
