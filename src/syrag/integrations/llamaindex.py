from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from syrag._optional import missing_optional_dependency
from syrag.protocols import Chunker
from syrag.schemas import DocumentChunk, SourceDocument

try:
    from llama_index.core import Document as LlamaIndexDocument
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via import path
    raise missing_optional_dependency(
        feature="syrag.integrations.llamaindex",
        extra="llamaindex",
    ) from exc

_SOURCE_ID_KEY = "syrag_source_id"
_PAGE_NUMBER_KEY = "syrag_page_number"


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
        content = self._content_for(node)
        metadata = self._metadata_from(node)
        source_id = self._source_id_for(metadata=metadata, node=node, index=index)
        page_number = self._page_number_for(metadata)
        metadata.pop(_SOURCE_ID_KEY, None)
        metadata.pop(_PAGE_NUMBER_KEY, None)

        return DocumentChunk(
            chunk_id=self._chunk_id_for(node=node, source_id=source_id, index=index),
            source_id=source_id,
            content=content,
            metadata=metadata,
            page_number=page_number,
            chunk_index=index,
        )

    def _content_for(self, node: Any) -> str:
        get_content = getattr(node, "get_content", None)
        if callable(get_content):
            content = get_content()
        else:
            content = getattr(node, "text", None)
        if not isinstance(content, str) or not content.strip():
            msg = "LlamaIndex node content must be a non-empty string"
            raise TypeError(msg)
        return content.strip()

    def _metadata_from(self, node: Any) -> dict[str, Any]:
        metadata = getattr(node, "metadata", {})
        if not isinstance(metadata, dict):
            return {}
        return dict(metadata)

    def _source_id_for(
        self,
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

    def _page_number_for(self, metadata: dict[str, Any]) -> int | None:
        page_number = metadata.get(_PAGE_NUMBER_KEY, metadata.get("page_number"))
        if isinstance(page_number, int):
            return page_number
        return None

    def _chunk_id_for(self, *, node: Any, source_id: str, index: int) -> str:
        node_id = getattr(node, "node_id", getattr(node, "id_", None))
        if isinstance(node_id, str) and node_id:
            return node_id
        return f"{source_id}-chunk-{index}"
