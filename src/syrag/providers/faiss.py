from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from syrag._optional import missing_optional_dependency
from syrag.protocols import EmbeddingVector, Filters, VectorStore
from syrag.schemas import DocumentChunk, RetrievedChunk

try:
    import faiss  # type: ignore[import-untyped]
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via import path
    raise missing_optional_dependency(
        feature="syrag.providers.faiss",
        extra="faiss",
    ) from exc


@dataclass(frozen=True)
class _StoredRecord:
    chunk: DocumentChunk
    embedding: list[float]
    collection: str
    tenant_id: str


class FAISSVectorStore(VectorStore):
    """FAISS-backed local vector store using cosine similarity."""

    def __init__(self, *, dimensions: int | None = None) -> None:
        self.dimensions = dimensions
        self._records: dict[tuple[str, str, str], _StoredRecord] = {}
        self._ordered_records: list[_StoredRecord] = []
        self._index: Any | None = self._create_index(dimensions)

    async def upsert(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
        collection: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        if len(chunks) != len(embeddings):
            msg = "chunks and embeddings must have the same length"
            raise ValueError(msg)
        if not chunks:
            return

        normalized_collection = self._namespace_key(collection)
        normalized_tenant = self._namespace_key(tenant_id)
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            vector = [float(value) for value in embedding]
            self._ensure_dimensions(vector)
            self._records[
                (normalized_collection, normalized_tenant, chunk.chunk_id)
            ] = _StoredRecord(
                chunk=chunk,
                embedding=vector,
                collection=normalized_collection,
                tenant_id=normalized_tenant,
            )

        self._rebuild_index()

    async def query(
        self,
        *,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: Filters | None = None,
    ) -> list[RetrievedChunk]:
        if self._index is None or not self._ordered_records:
            return []

        query_vector = [float(value) for value in query_embedding]
        self._ensure_dimensions(query_vector)
        distances, indices = self._index.search(
            self._normalized_matrix([query_vector]),
            len(self._ordered_records),
        )
        normalized_collection = self._namespace_key(collection)
        normalized_tenant = self._namespace_key(tenant_id)

        results: list[RetrievedChunk] = []
        for score, record_index in zip(distances[0], indices[0], strict=True):
            if int(record_index) < 0:
                continue
            record = self._ordered_records[int(record_index)]
            if not self._matches_namespace(record, normalized_collection, normalized_tenant):
                continue
            if not self._matches_filters(record.chunk.metadata, filters):
                continue
            results.append(self._retrieved_chunk_for(record, score=float(score)))
            if len(results) >= top_k:
                break
        return results

    def _create_index(self, dimensions: int | None) -> Any | None:
        if dimensions is None:
            return None
        return faiss.IndexFlatIP(dimensions)

    def _ensure_dimensions(self, vector: list[float]) -> None:
        if not vector:
            msg = "FAISS embeddings must not be empty"
            raise ValueError(msg)
        if self.dimensions is None:
            self.dimensions = len(vector)
            self._index = self._create_index(self.dimensions)
        if len(vector) != self.dimensions:
            msg = f"expected embedding dimension {self.dimensions}, received {len(vector)}"
            raise ValueError(msg)

    def _rebuild_index(self) -> None:
        if self.dimensions is None:
            return
        self._ordered_records = list(self._records.values())
        self._index = self._create_index(self.dimensions)
        if self._index is not None and self._ordered_records:
            self._index.add(
                self._normalized_matrix([record.embedding for record in self._ordered_records])
            )

    def _normalized_matrix(self, vectors: list[list[float]]) -> Any:
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        return matrix

    def _retrieved_chunk_for(self, record: _StoredRecord, *, score: float) -> RetrievedChunk:
        chunk = record.chunk
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            content=chunk.content,
            score=self._bounded_cosine_score(score),
            metadata=chunk.metadata,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
        )

    def _matches_namespace(
        self,
        record: _StoredRecord,
        collection: str,
        tenant_id: str,
    ) -> bool:
        return record.collection == collection and record.tenant_id == tenant_id

    def _matches_filters(self, metadata: dict[str, Any], filters: Filters | None) -> bool:
        return all(metadata.get(key) == value for key, value in (filters or {}).items())

    def _bounded_cosine_score(self, score: float) -> float:
        return max(0.0, min(1.0, score))

    def _namespace_key(self, value: str | None) -> str:
        return value or ""
