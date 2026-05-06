from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from syrag._optional import missing_optional_dependency
from syrag.protocols import Chunker, EmbeddingVector, VectorStore
from syrag.schemas import DocumentChunk, QueryRequest, RetrievedChunk, SourceDocument

try:
    import langchain_text_splitters  # noqa: F401
    from langchain_core.documents import Document
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via import path
    raise missing_optional_dependency(
        feature="syrag.integrations.langchain",
        extra="langchain",
    ) from exc


class LangChainTextChunker(Chunker):
    """Adapts a LangChain text splitter to the SyRAG Chunker protocol."""

    def __init__(self, *, text_splitter: Any) -> None:
        split_text = getattr(text_splitter, "split_text", None)
        if not callable(split_text):
            msg = "text_splitter must expose a callable split_text(text) method"
            raise TypeError(msg)
        self.text_splitter = text_splitter

    async def chunk(
        self,
        documents: Sequence[SourceDocument],
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for document in documents:
            split_texts = self._split_document(document)
            chunks.extend(
                DocumentChunk(
                    chunk_id=f"{document.source_id}-chunk-{chunk_index}",
                    source_id=document.source_id,
                    content=content,
                    metadata=dict(document.metadata),
                    page_number=document.page_number,
                    chunk_index=chunk_index,
                )
                for chunk_index, content in enumerate(split_texts)
            )
        return chunks

    def _split_document(self, document: SourceDocument) -> list[str]:
        raw_chunks = self.text_splitter.split_text(document.content)
        if not isinstance(raw_chunks, Sequence) or isinstance(raw_chunks, (str, bytes)):
            msg = "text_splitter.split_text(text) must return a sequence of strings"
            raise TypeError(msg)

        chunks: list[str] = []
        for raw_chunk in raw_chunks:
            if not isinstance(raw_chunk, str):
                msg = "text_splitter.split_text(text) must return only strings"
                raise TypeError(msg)
            chunk = raw_chunk.strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class LangChainRetrieverStrategy:
    """Adapts a LangChain retriever to SyRAG query retrieval."""

    def __init__(self, *, retriever: Any) -> None:
        ainvoke = getattr(retriever, "ainvoke", None)
        invoke = getattr(retriever, "invoke", None)
        if not callable(ainvoke) and not callable(invoke):
            msg = "retriever must expose callable invoke(query) or ainvoke(query) methods"
            raise TypeError(msg)
        self.retriever = retriever

    async def retrieve(
        self,
        *,
        request: QueryRequest,
        query_embedding: EmbeddingVector,
        vector_store: VectorStore,
    ) -> list[RetrievedChunk]:
        del query_embedding, vector_store
        documents = await self._retrieve_documents(request.query)
        return [
            self._retrieved_chunk_for(document, index=index)
            for index, document in enumerate(documents[: request.top_k])
        ]

    async def _retrieve_documents(self, query: str) -> list[Document]:
        ainvoke = getattr(self.retriever, "ainvoke", None)
        if callable(ainvoke):
            raw_documents = await ainvoke(query)
        else:
            invoke = self.retriever.invoke
            raw_documents = invoke(query)

        if inspect.isawaitable(raw_documents):
            raw_documents = await raw_documents
        if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, (str, bytes)):
            msg = "retriever must return a sequence of LangChain Document objects"
            raise TypeError(msg)
        return [self._validate_document(document) for document in raw_documents]

    def _validate_document(self, document: Any) -> Document:
        if not isinstance(document, Document):
            msg = "retriever must return LangChain Document objects"
            raise TypeError(msg)
        return document

    def _retrieved_chunk_for(self, document: Document, *, index: int) -> RetrievedChunk:
        metadata = dict(document.metadata)
        source_id = self._string_metadata_value(
            metadata,
            keys=("source_id", "source", "id"),
            default=f"langchain-{index}",
        )
        chunk_index = self._int_metadata_value(metadata, keys=("chunk_index",), default=index)
        return RetrievedChunk(
            chunk_id=self._string_metadata_value(
                metadata,
                keys=("chunk_id", "id"),
                default=f"{source_id}-chunk-{chunk_index}",
            ),
            source_id=source_id,
            content=document.page_content,
            score=self._score_for(metadata),
            metadata=metadata,
            page_number=self._optional_int_metadata_value(
                metadata,
                keys=("page_number", "page"),
            ),
            chunk_index=chunk_index,
        )

    def _score_for(self, metadata: dict[str, Any]) -> float:
        raw_score = metadata.get("score", metadata.get("relevance_score", metadata.get("_score")))
        if isinstance(raw_score, int | float):
            return max(0.0, min(1.0, float(raw_score)))
        return 1.0

    def _string_metadata_value(
        self,
        metadata: dict[str, Any],
        *,
        keys: Sequence[str],
        default: str,
    ) -> str:
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
        return default

    def _int_metadata_value(
        self,
        metadata: dict[str, Any],
        *,
        keys: Sequence[str],
        default: int,
    ) -> int:
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, int):
                return value
        return default

    def _optional_int_metadata_value(
        self,
        metadata: dict[str, Any],
        *,
        keys: Sequence[str],
    ) -> int | None:
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, int):
                return value
        return None
