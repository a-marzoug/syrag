from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, cast

from pydantic import BaseModel, Field

from syrag._optional import missing_optional_dependency
from syrag.protocols import Chunker, EmbeddingVector, VectorStore
from syrag.schemas import (
    Citation,
    DocumentChunk,
    QueryRequest,
    RAGResponse,
    RetrievedChunk,
    SourceDocument,
)

try:
    import httpx
    import langchain_text_splitters  # noqa: F401
    from langchain_core.documents import Document
    from langchain_core.tools import StructuredTool
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via import path
    raise missing_optional_dependency(
        feature="syrag.integrations.langchain",
        extra="langchain",
    ) from exc


class SyRAGQueryToolInput(BaseModel):
    """Input schema for LangChain tools that call a SyRAG query endpoint."""

    query: str = Field(description="Question to answer with the SyRAG query endpoint.")
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of retrieved chunks SyRAG should use.",
    )
    collection: str | None = Field(
        default=None,
        description="Optional SyRAG collection to query.",
    )
    tenant_id: str | None = Field(
        default=None,
        description="Optional tenant identifier when the SyRAG service is tenant-scoped.",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata filters passed to the SyRAG query endpoint.",
    )


def create_syrag_query_tool(
    *,
    base_url: str,
    path: str = "/query",
    name: str = "query_syrag",
    description: str = (
        "Query a SyRAG RAG service for grounded answers and cited context."
    ),
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 30.0,
    transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
) -> StructuredTool:
    """Create a LangChain tool that calls a SyRAG query endpoint."""

    endpoint = _endpoint_url(base_url=base_url, path=path)
    sync_transport = cast(httpx.BaseTransport | None, transport)
    async_transport = cast(httpx.AsyncBaseTransport | None, transport)

    def query_syrag(
        query: str,
        top_k: int = 5,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """Query a SyRAG RAG service for grounded answers."""

        with httpx.Client(
            headers=headers,
            timeout=timeout_seconds,
            transport=sync_transport,
        ) as client:
            response = client.post(
                endpoint,
                json=_query_payload(
                    query=query,
                    top_k=top_k,
                    collection=collection,
                    tenant_id=tenant_id,
                    filters=filters,
                ),
            )
            response.raise_for_status()
            return _format_query_response(RAGResponse.model_validate(response.json()))

    async def aquery_syrag(
        query: str,
        top_k: int = 5,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """Query a SyRAG RAG service for grounded answers."""

        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout_seconds,
            transport=async_transport,
        ) as client:
            response = await client.post(
                endpoint,
                json=_query_payload(
                    query=query,
                    top_k=top_k,
                    collection=collection,
                    tenant_id=tenant_id,
                    filters=filters,
                ),
            )
            response.raise_for_status()
            return _format_query_response(RAGResponse.model_validate(response.json()))

    return StructuredTool.from_function(
        func=query_syrag,
        coroutine=aquery_syrag,
        name=name,
        description=description,
        args_schema=SyRAGQueryToolInput,
    )


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


def _endpoint_url(*, base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _query_payload(
    *,
    query: str,
    top_k: int,
    collection: str | None,
    tenant_id: str | None,
    filters: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "filters": filters or {},
    }
    if collection is not None:
        payload["collection"] = collection
    if tenant_id is not None:
        payload["tenant_id"] = tenant_id
    return payload


def _format_query_response(response: RAGResponse) -> str:
    if not response.citations:
        return response.answer

    citations = "\n".join(
        _format_citation(citation, index=index)
        for index, citation in enumerate(response.citations, start=1)
    )
    return f"{response.answer}\n\nSources:\n{citations}"


def _format_citation(citation: Citation, *, index: int) -> str:
    page_suffix = f", page {citation.page_number}" if citation.page_number is not None else ""
    return (
        f"{index}. {citation.source_id}{page_suffix} "
        f"(score: {citation.score:.2f}) - {citation.snippet}"
    )
