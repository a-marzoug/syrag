from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fastrag.protocols import LLM, Embedder, EmbeddingVector, Filters, Metadata, VectorStore
from fastrag.schemas import Citation, QueryRequest, RAGResponse, RetrievedDocument

TOKEN_PATTERN = re.compile(r"\b\w+\b")


@dataclass(slots=True)
class StoredDocument:
    source_id: str
    content: str
    embedding: tuple[float, ...]
    metadata: dict[str, Any]
    page_number: int | None


class InMemoryEmbedder(Embedder):
    """Deterministic hash-based embedder for local development and tests."""

    def __init__(self, dimensions: int = 16) -> None:
        if dimensions <= 0:
            msg = "dimensions must be a positive integer"
            raise ValueError(msg)
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            bucket = self._bucket_for_token(token)
            vector[bucket] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            return vector

        return [value / magnitude for value in vector]

    def _bucket_for_token(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big") % self.dimensions


class InMemoryVectorStore(VectorStore):
    """Collection- and tenant-aware in-memory vector store."""

    def __init__(self) -> None:
        self._namespaces: dict[tuple[str | None, str | None], dict[str, StoredDocument]] = {}

    async def upsert(
        self,
        *,
        documents: Sequence[str],
        embeddings: Sequence[EmbeddingVector],
        collection: str | None = None,
        tenant_id: str | None = None,
        metadata: Sequence[Metadata] | None = None,
    ) -> None:
        if len(documents) != len(embeddings):
            msg = "documents and embeddings must have the same length"
            raise ValueError(msg)

        if metadata is not None and len(metadata) != len(documents):
            msg = "metadata must match the number of documents"
            raise ValueError(msg)

        namespace = self._namespaces.setdefault((collection, tenant_id), {})
        resolved_metadata = metadata or [{} for _ in documents]

        for index, (document, embedding, item_metadata) in enumerate(
            zip(documents, embeddings, resolved_metadata, strict=True)
        ):
            source_id = self._resolve_source_id(item_metadata, index)
            page_number = self._coerce_page_number(item_metadata.get("page_number"))
            namespace[source_id] = StoredDocument(
                source_id=source_id,
                content=document,
                embedding=tuple(float(value) for value in embedding),
                metadata=dict(item_metadata),
                page_number=page_number,
            )

    async def query(
        self,
        *,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: Filters | None = None,
    ) -> list[RetrievedDocument]:
        namespace = self._namespaces.get((collection, tenant_id), {})
        normalized_query = tuple(float(value) for value in query_embedding)
        candidates = [
            document
            for document in namespace.values()
            if self._matches_filters(document.metadata, filters)
        ]
        ranked_documents = sorted(
            candidates,
            key=lambda document: self._cosine_similarity(normalized_query, document.embedding),
            reverse=True,
        )

        return [
            RetrievedDocument(
                source_id=document.source_id,
                content=document.content,
                score=self._cosine_similarity(normalized_query, document.embedding),
                metadata=document.metadata,
                page_number=document.page_number,
            )
            for document in ranked_documents[:top_k]
        ]

    def _resolve_source_id(self, metadata: Mapping[str, Any], index: int) -> str:
        raw_source_id = metadata.get("source_id")
        if isinstance(raw_source_id, str) and raw_source_id.strip():
            return raw_source_id.strip()
        return f"doc-{index}"

    def _coerce_page_number(self, raw_page_number: Any) -> int | None:
        if raw_page_number is None:
            return None
        if isinstance(raw_page_number, int) and raw_page_number > 0:
            return raw_page_number
        return None

    def _matches_filters(
        self,
        metadata: Mapping[str, Any],
        filters: Filters | None,
    ) -> bool:
        if not filters:
            return True
        return all(metadata.get(key) == value for key, value in filters.items())

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


class InMemoryLLM(LLM):
    """Simple grounded generator for development and integration tests."""

    def __init__(self, max_context_documents: int = 3) -> None:
        if max_context_documents <= 0:
            msg = "max_context_documents must be a positive integer"
            raise ValueError(msg)
        self.max_context_documents = max_context_documents

    async def generate(
        self,
        *,
        query: QueryRequest,
        context: Sequence[RetrievedDocument],
    ) -> RAGResponse:
        limited_context = list(context[: self.max_context_documents])
        if not limited_context:
            return RAGResponse(
                answer=f"No grounded context was available for query: {query.query}",
                citations=[],
                usage=self._usage_for(query.query, ""),
            )

        supporting_passages = " ".join(document.content for document in limited_context)
        answer = f"Grounded answer for '{query.query}': {supporting_passages}"

        return RAGResponse(
            answer=answer,
            citations=[
                Citation(
                    source_id=document.source_id,
                    score=document.score,
                    snippet=document.content,
                    page_number=document.page_number,
                )
                for document in limited_context
            ],
            usage=self._usage_for(query.query, answer),
        )

    def _usage_for(self, prompt: str, answer: str) -> dict[str, int]:
        return {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(answer.split()),
        }
