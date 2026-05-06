from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from syrag._optional import missing_optional_dependency
from syrag.protocols import Chunker
from syrag.schemas import DocumentChunk, SourceDocument

try:
    import langchain_text_splitters  # noqa: F401
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
