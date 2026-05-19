from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from syrag._optional import missing_optional_dependency
from syrag.protocols import EmbeddingVector, Filters, VectorStore
from syrag.schemas import DocumentChunk, RetrievedChunk

try:
    from qdrant_client import QdrantClient, models
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via import path
    raise missing_optional_dependency(
        feature="syrag.providers.qdrant",
        extra="qdrant",
    ) from exc

_COLLECTION_KEY = "syrag_collection"
_TENANT_KEY = "syrag_tenant"
_SOURCE_ID_KEY = "syrag_source_id"
_PAGE_NUMBER_KEY = "syrag_page_number"
_CHUNK_INDEX_KEY = "syrag_chunk_index"
_CHUNK_ID_KEY = "syrag_chunk_id"
_CONTENT_KEY = "syrag_content"
_METADATA_JSON_KEY = "syrag_metadata_json"
_FILTER_PREFIX = "meta__"


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store for local or remote Qdrant deployments."""

    def __init__(
        self,
        *,
        collection_name: str = "syrag_documents",
        dimensions: int | None = None,
        path: str | Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        distance: Any | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.path = Path(path).expanduser() if path is not None else None
        self.url = url
        self.api_key = api_key
        self.distance = distance or models.Distance.COSINE
        self.client = client or self._create_client()
        self._collection_ready = False

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

        vectors = [[float(value) for value in embedding] for embedding in embeddings]
        self._ensure_dimensions(vectors[0])
        self._ensure_collection()

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=self._point_id(chunk, collection, tenant_id),
                    vector=vector,
                    payload=self._payload_for(
                        chunk,
                        collection=collection,
                        tenant_id=tenant_id,
                    ),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
            wait=True,
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
        if top_k <= 0:
            return []
        query_vector = [float(value) for value in query_embedding]
        self._ensure_dimensions(query_vector)
        if not self._collection_ready and not self._collection_exists():
            return []
        self._collection_ready = True

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._filter_for(
                collection=collection,
                tenant_id=tenant_id,
                filters=filters,
            ),
            limit=top_k,
            with_payload=True,
        )
        return [self._retrieved_chunk_for(point) for point in result.points]

    def _create_client(self) -> Any:
        if self.url is not None:
            return QdrantClient(url=self.url, api_key=self.api_key)
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=str(self.path))
        return QdrantClient(":memory:")

    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        if self._collection_exists():
            self._collection_ready = True
            return
        if self.dimensions is None:
            msg = "Qdrant vector dimensions must be known before collection creation"
            raise ValueError(msg)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimensions,
                distance=self.distance,
            ),
        )
        self._collection_ready = True

    def _collection_exists(self) -> bool:
        collection_exists = getattr(self.client, "collection_exists", None)
        if callable(collection_exists):
            return bool(collection_exists(collection_name=self.collection_name))
        return False

    def _ensure_dimensions(self, vector: list[float]) -> None:
        if not vector:
            msg = "Qdrant embeddings must not be empty"
            raise ValueError(msg)
        if self.dimensions is None:
            self.dimensions = len(vector)
        if len(vector) != self.dimensions:
            msg = f"expected embedding dimension {self.dimensions}, received {len(vector)}"
            raise ValueError(msg)

    def _point_id(
        self,
        chunk: DocumentChunk,
        collection: str | None,
        tenant_id: str | None,
    ) -> str:
        raw_id = ":".join(
            [
                self.collection_name,
                self._namespace_key(collection),
                self._namespace_key(tenant_id),
                chunk.chunk_id,
            ]
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))

    def _payload_for(
        self,
        chunk: DocumentChunk,
        *,
        collection: str | None,
        tenant_id: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            _COLLECTION_KEY: self._namespace_key(collection),
            _TENANT_KEY: self._namespace_key(tenant_id),
            _SOURCE_ID_KEY: chunk.source_id,
            _CHUNK_ID_KEY: chunk.chunk_id,
            _CONTENT_KEY: chunk.content,
            _CHUNK_INDEX_KEY: chunk.chunk_index,
            _METADATA_JSON_KEY: json.dumps(chunk.metadata),
        }
        if chunk.page_number is not None:
            payload[_PAGE_NUMBER_KEY] = chunk.page_number
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                payload[f"{_FILTER_PREFIX}{key}"] = value
        return payload

    def _filter_for(
        self,
        *,
        collection: str | None,
        tenant_id: str | None,
        filters: Filters | None,
    ) -> Any:
        conditions = [
            self._match_condition(_COLLECTION_KEY, self._namespace_key(collection)),
            self._match_condition(_TENANT_KEY, self._namespace_key(tenant_id)),
        ]
        for key, value in (filters or {}).items():
            if isinstance(value, (str, int, bool)):
                conditions.append(self._match_condition(f"{_FILTER_PREFIX}{key}", value))
        return models.Filter(must=conditions)

    def _match_condition(self, key: str, value: str | int | bool) -> Any:
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    def _retrieved_chunk_for(self, point: Any) -> RetrievedChunk:
        payload = self._payload_for_point(point)
        chunk_id = str(payload.get(_CHUNK_ID_KEY, point.id))
        return RetrievedChunk(
            chunk_id=chunk_id,
            source_id=str(payload.get(_SOURCE_ID_KEY, chunk_id)),
            content=str(payload.get(_CONTENT_KEY, "")),
            score=self._bounded_score(float(getattr(point, "score", 0.0))),
            metadata=self._stored_metadata_for(payload),
            page_number=self._optional_int(payload.get(_PAGE_NUMBER_KEY)),
            chunk_index=int(payload.get(_CHUNK_INDEX_KEY, 0)),
        )

    def _payload_for_point(self, point: Any) -> Mapping[str, Any]:
        payload = getattr(point, "payload", None)
        if isinstance(payload, Mapping):
            return payload
        return {}

    def _stored_metadata_for(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raw_metadata = payload.get(_METADATA_JSON_KEY)
        if not isinstance(raw_metadata, str):
            return {}
        try:
            parsed_metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed_metadata, dict):
            return {}
        return dict(parsed_metadata)

    def _bounded_score(self, score: float) -> float:
        if not math.isfinite(score):
            return 0.0
        return max(0.0, min(1.0, score))

    def _namespace_key(self, value: str | None) -> str:
        return value or ""

    def _optional_int(self, value: Any) -> int | None:
        if value is None:
            return None
        return int(value)
