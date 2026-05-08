from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, cast

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
    from llama_index.core import Document as LlamaIndexDocument
    from llama_index.core.base.response.schema import Response
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via import path
    raise missing_optional_dependency(
        feature="syrag.integrations.llamaindex",
        extra="llamaindex",
    ) from exc

_SOURCE_ID_KEY = "syrag_source_id"
_PAGE_NUMBER_KEY = "syrag_page_number"


class SyRAGQueryEngine(BaseQueryEngine):
    """LlamaIndex query engine that delegates retrieval and generation to SyRAG."""

    def __init__(
        self,
        *,
        base_url: str,
        path: str = "/query",
        top_k: int = 5,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(callback_manager=None)
        self.endpoint = _endpoint_url(base_url=base_url, path=path)
        self.top_k = top_k
        self.collection = collection
        self.tenant_id = tenant_id
        self.filters = filters or {}
        self.headers = headers
        self.timeout_seconds = timeout_seconds
        self.sync_transport = cast(httpx.BaseTransport | None, transport)
        self.async_transport = cast(httpx.AsyncBaseTransport | None, transport)

    def _query(self, query_bundle: QueryBundle) -> Response:
        with httpx.Client(
            headers=self.headers,
            timeout=self.timeout_seconds,
            transport=self.sync_transport,
        ) as client:
            response = client.post(
                self.endpoint,
                json=self._query_payload(query_bundle.query_str),
            )
            response.raise_for_status()
            return _response_for(RAGResponse.model_validate(response.json()))

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        async with httpx.AsyncClient(
            headers=self.headers,
            timeout=self.timeout_seconds,
            transport=self.async_transport,
        ) as client:
            response = await client.post(
                self.endpoint,
                json=self._query_payload(query_bundle.query_str),
            )
            response.raise_for_status()
            return _response_for(RAGResponse.model_validate(response.json()))

    def _get_prompt_modules(self) -> dict[str, Any]:
        return {}

    def _query_payload(self, query: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "top_k": self.top_k,
            "filters": self.filters,
        }
        if self.collection is not None:
            payload["collection"] = self.collection
        if self.tenant_id is not None:
            payload["tenant_id"] = self.tenant_id
        return payload


class LlamaIndexNodeChunker(Chunker):
    """Adapts a LlamaIndex node parser to the SyRAG Chunker protocol."""

    def __init__(self, *, node_parser: Any) -> None:
        get_nodes_from_documents = getattr(node_parser, "get_nodes_from_documents", None)
        if not callable(get_nodes_from_documents):
            msg = (
                "node_parser must expose a callable "
                "get_nodes_from_documents(documents) method"
            )
            raise TypeError(msg)
        self.node_parser = node_parser

    async def chunk(
        self,
        documents: Sequence[SourceDocument],
    ) -> list[DocumentChunk]:
        nodes = self._parse_documents(documents)
        return [
            self._document_chunk_for(node, index=index)
            for index, node in enumerate(nodes)
        ]

    def _parse_documents(self, documents: Sequence[SourceDocument]) -> list[Any]:
        llama_documents = [
            LlamaIndexDocument(
                text=document.content,
                metadata=self._metadata_for(document),
            )
            for document in documents
        ]
        get_nodes_from_documents = self.node_parser.get_nodes_from_documents
        signature = inspect.signature(get_nodes_from_documents)
        if "show_progress" in signature.parameters:
            raw_nodes = get_nodes_from_documents(llama_documents, show_progress=False)
        else:
            raw_nodes = get_nodes_from_documents(llama_documents)

        if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)):
            msg = "node_parser.get_nodes_from_documents(...) must return a sequence of nodes"
            raise TypeError(msg)
        return list(raw_nodes)

    def _metadata_for(self, document: SourceDocument) -> dict[str, Any]:
        metadata = dict(document.metadata)
        metadata[_SOURCE_ID_KEY] = document.source_id
        if document.page_number is not None:
            metadata[_PAGE_NUMBER_KEY] = document.page_number
        return metadata

    def _document_chunk_for(self, node: Any, *, index: int) -> DocumentChunk:
        content = _content_for(node)
        metadata = _metadata_from(node)
        source_id = _source_id_for(metadata=metadata, node=node, index=index)
        page_number = _page_number_for(metadata)
        metadata.pop(_SOURCE_ID_KEY, None)
        metadata.pop(_PAGE_NUMBER_KEY, None)

        return DocumentChunk(
            chunk_id=_chunk_id_for(node=node, source_id=source_id, index=index),
            source_id=source_id,
            content=content,
            metadata=metadata,
            page_number=page_number,
            chunk_index=index,
        )


class LlamaIndexRetrieverStrategy:
    """Adapts a LlamaIndex retriever to SyRAG query retrieval."""

    def __init__(self, *, retriever: Any) -> None:
        aretrieve = getattr(retriever, "aretrieve", None)
        retrieve = getattr(retriever, "retrieve", None)
        if not callable(aretrieve) and not callable(retrieve):
            msg = (
                "retriever must expose callable retrieve(query) or "
                "aretrieve(query) methods"
            )
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
        nodes = await self._retrieve_nodes(request.query)
        return [
            self._retrieved_chunk_for(node_with_score, index=index)
            for index, node_with_score in enumerate(nodes[: request.top_k])
        ]

    async def _retrieve_nodes(self, query: str) -> list[Any]:
        aretrieve = getattr(self.retriever, "aretrieve", None)
        if callable(aretrieve):
            raw_nodes = await aretrieve(query)
        else:
            retrieve = self.retriever.retrieve
            raw_nodes = retrieve(query)

        if inspect.isawaitable(raw_nodes):
            raw_nodes = await raw_nodes
        if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)):
            msg = "retriever must return a sequence of LlamaIndex NodeWithScore objects"
            raise TypeError(msg)
        return list(raw_nodes)

    def _retrieved_chunk_for(self, node_with_score: Any, *, index: int) -> RetrievedChunk:
        node = getattr(node_with_score, "node", None)
        if node is None:
            msg = "LlamaIndex retrieval result must expose a node attribute"
            raise TypeError(msg)

        content = _content_for(node)
        metadata = _metadata_from(node)
        source_id = _source_id_for(metadata=metadata, node=node, index=index)
        page_number = _page_number_for(metadata)
        metadata.pop(_SOURCE_ID_KEY, None)
        metadata.pop(_PAGE_NUMBER_KEY, None)

        return RetrievedChunk(
            chunk_id=_chunk_id_for(node=node, source_id=source_id, index=index),
            source_id=source_id,
            content=content,
            score=self._score_for(node_with_score),
            metadata=metadata,
            page_number=page_number,
            chunk_index=self._chunk_index_for(metadata=metadata, index=index),
        )

    def _score_for(self, node_with_score: Any) -> float:
        get_score = getattr(node_with_score, "get_score", None)
        if callable(get_score):
            score = get_score()
        else:
            score = getattr(node_with_score, "score", None)
        if isinstance(score, int | float):
            return max(0.0, min(1.0, float(score)))
        return 1.0

    def _chunk_index_for(self, *, metadata: dict[str, Any], index: int) -> int:
        chunk_index = metadata.get("chunk_index")
        if isinstance(chunk_index, int):
            return chunk_index
        return index


def _content_for(node: Any) -> str:
    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        content = get_content()
    else:
        content = getattr(node, "text", None)
    if not isinstance(content, str) or not content.strip():
        msg = "LlamaIndex node content must be a non-empty string"
        raise TypeError(msg)
    return content.strip()


def _metadata_from(node: Any) -> dict[str, Any]:
    metadata = getattr(node, "metadata", {})
    if not isinstance(metadata, dict):
        return {}
    return dict(metadata)


def _source_id_for(
    *,
    metadata: dict[str, Any],
    node: Any,
    index: int,
) -> str:
    metadata_source_id = metadata.get(_SOURCE_ID_KEY, metadata.get("source_id"))
    if isinstance(metadata_source_id, str) and metadata_source_id:
        return metadata_source_id

    source_node = getattr(node, "source_node", None)
    source_node_id = getattr(source_node, "node_id", None)
    if isinstance(source_node_id, str) and source_node_id:
        return source_node_id

    return f"llamaindex-{index}"


def _page_number_for(metadata: dict[str, Any]) -> int | None:
    page_number = metadata.get(_PAGE_NUMBER_KEY, metadata.get("page_number"))
    if isinstance(page_number, int):
        return page_number
    return None


def _chunk_id_for(*, node: Any, source_id: str, index: int) -> str:
    node_id = getattr(node, "node_id", getattr(node, "id_", None))
    if isinstance(node_id, str) and node_id:
        return node_id
    return f"{source_id}-chunk-{index}"


def _response_for(response: RAGResponse) -> Response:
    return Response(
        response=response.answer,
        source_nodes=[
            _node_with_score_for(citation, index=index)
            for index, citation in enumerate(response.citations)
        ],
        metadata={
            "usage": response.usage,
            "citations": [citation.model_dump() for citation in response.citations],
        },
    )


def _node_with_score_for(citation: Citation, *, index: int) -> NodeWithScore:
    metadata: dict[str, Any] = {"source_id": citation.source_id}
    if citation.page_number is not None:
        metadata["page_number"] = citation.page_number
    return NodeWithScore(
        node=TextNode(
            id_=f"{citation.source_id}-citation-{index}",
            text=citation.snippet,
            metadata=metadata,
        ),
        score=citation.score,
    )


def _endpoint_url(*, base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"
