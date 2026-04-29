from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from syrag._optional import missing_optional_dependency
from syrag.protocols import EmbeddingVector, Filters, VectorStore
from syrag.schemas import DocumentChunk, RetrievedChunk

try:
    import chromadb
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via import path
    raise missing_optional_dependency(
        feature="syrag.providers.chroma",
        extra="chroma",
    ) from exc

_COLLECTION_KEY = "syrag_collection"
_TENANT_KEY = "syrag_tenant"
_SOURCE_ID_KEY = "syrag_source_id"
_PAGE_NUMBER_KEY = "syrag_page_number"
_CHUNK_INDEX_KEY = "syrag_chunk_index"
_METADATA_JSON_KEY = "syrag_metadata_json"
_FILTER_PREFIX = "meta__"


class ChromaVectorStore(VectorStore):
    """Chroma-backed vector store for local development and Chroma deployments."""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        collection_name: str = "syrag_documents",
        client: Any | None = None,
    ) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self.collection_name = collection_name
        self.client = client or self._create_client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )

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

        self.collection.upsert(
            ids=[self._record_id(chunk, collection, tenant_id) for chunk in chunks],
            embeddings=[
                [float(value) for value in embedding]
                for embedding in embeddings
            ],
            metadatas=[
                self._metadata_for(chunk, collection=collection, tenant_id=tenant_id)
                for chunk in chunks
            ],
            documents=[chunk.content for chunk in chunks],
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
        result = self.collection.query(
            query_embeddings=[[float(value) for value in query_embedding]],
            n_results=top_k,
            where=self._where_for(collection=collection, tenant_id=tenant_id, filters=filters),
            include=["documents", "metadatas", "distances"],
        )
        return self._retrieved_chunks_for(result)

    def _create_client(self) -> Any:
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            return chromadb.PersistentClient(path=str(self.path))
        return chromadb.EphemeralClient()

    def _record_id(
        self,
        chunk: DocumentChunk,
        collection: str | None,
        tenant_id: str | None,
    ) -> str:
        namespace = f"{self._namespace_key(collection)}:{self._namespace_key(tenant_id)}"
        return f"{self._slug(namespace)}:{chunk.chunk_id}"

    def _metadata_for(
        self,
        chunk: DocumentChunk,
        *,
        collection: str | None,
        tenant_id: str | None,
    ) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {
            _COLLECTION_KEY: self._namespace_key(collection),
            _TENANT_KEY: self._namespace_key(tenant_id),
            _SOURCE_ID_KEY: chunk.source_id,
            _CHUNK_INDEX_KEY: chunk.chunk_index,
            _METADATA_JSON_KEY: json.dumps(chunk.metadata),
        }
        if chunk.page_number is not None:
            metadata[_PAGE_NUMBER_KEY] = chunk.page_number
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[f"{_FILTER_PREFIX}{key}"] = value
        return metadata

    def _where_for(
        self,
        *,
        collection: str | None,
        tenant_id: str | None,
        filters: Filters | None,
    ) -> dict[str, Any]:
        predicates: list[dict[str, Any]] = [
            {_COLLECTION_KEY: self._namespace_key(collection)},
            {_TENANT_KEY: self._namespace_key(tenant_id)},
        ]
        for key, value in (filters or {}).items():
            if isinstance(value, (str, int, float, bool)):
                predicates.append({f"{_FILTER_PREFIX}{key}": value})

        if len(predicates) == 1:
            return predicates[0]
        return {"$and": predicates}

    def _retrieved_chunks_for(self, result: Mapping[str, Any]) -> list[RetrievedChunk]:
        ids = self._first_result_list(result.get("ids"))
        documents = self._first_result_list(result.get("documents"))
        metadatas = self._first_result_list(result.get("metadatas"))
        distances = self._first_result_list(result.get("distances"))

        chunks: list[RetrievedChunk] = []
        for index, record_id in enumerate(ids):
            metadata = self._metadata_at(metadatas, index)
            document = self._value_at(documents, index, "")
            distance = float(self._value_at(distances, index, 0.0))
            chunk_id = str(record_id).rsplit(":", maxsplit=1)[-1]
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source_id=str(metadata.get(_SOURCE_ID_KEY, chunk_id)),
                    content=str(document),
                    score=self._score_for_distance(distance),
                    metadata=self._stored_metadata_for(metadata),
                    page_number=self._optional_int(metadata.get(_PAGE_NUMBER_KEY)),
                    chunk_index=int(metadata.get(_CHUNK_INDEX_KEY, 0)),
                )
            )
        return chunks

    def _stored_metadata_for(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        raw_metadata = metadata.get(_METADATA_JSON_KEY)
        if not isinstance(raw_metadata, str):
            return {}
        try:
            parsed_metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed_metadata, dict):
            return {}
        return dict(parsed_metadata)

    def _score_for_distance(self, distance: float) -> float:
        if not math.isfinite(distance):
            return 0.0
        return max(0.0, min(1.0, 1.0 / (1.0 + max(distance, 0.0))))

    def _namespace_key(self, value: str | None) -> str:
        return value or ""

    def _slug(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", value)
        return slug or "default"

    def _first_result_list(self, value: Any) -> list[Any]:
        if not isinstance(value, list) or not value:
            return []
        first_value = value[0]
        if not isinstance(first_value, list):
            return []
        return first_value

    def _metadata_at(self, metadatas: Sequence[Any], index: int) -> Mapping[str, Any]:
        metadata = self._value_at(metadatas, index, {})
        if isinstance(metadata, Mapping):
            return metadata
        return {}

    def _value_at(self, values: Sequence[Any], index: int, default: Any) -> Any:
        if index >= len(values):
            return default
        value = values[index]
        return default if value is None else value

    def _optional_int(self, value: Any) -> int | None:
        if value is None:
            return None
        return int(value)
