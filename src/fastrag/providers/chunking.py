from __future__ import annotations

from collections.abc import Sequence

from fastrag.protocols import Chunker
from fastrag.schemas import DocumentChunk, SourceDocument


class PassThroughChunker(Chunker):
    """Default chunker that emits one chunk per source document."""

    async def chunk(
        self,
        documents: Sequence[SourceDocument],
    ) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_id=f"{document.source_id}-chunk-0",
                source_id=document.source_id,
                content=document.content,
                metadata=dict(document.metadata),
                page_number=document.page_number,
                chunk_index=0,
            )
            for document in documents
        ]
