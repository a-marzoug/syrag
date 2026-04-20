import pytest

from fastrag.providers import PassThroughChunker
from fastrag.schemas import SourceDocument


@pytest.mark.asyncio
async def test_pass_through_chunker_emits_one_chunk_per_source_document() -> None:
    chunker = PassThroughChunker()

    chunks = await chunker.chunk(
        [
            SourceDocument(
                source_id="doc-1",
                content="FastRAG source content.",
                metadata={"topic": "framework"},
                page_number=1,
            )
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc-1-chunk-0"
    assert chunks[0].source_id == "doc-1"
    assert chunks[0].content == "FastRAG source content."
