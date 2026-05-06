import pytest

from syrag.integrations.langchain import LangChainTextSplitterChunker
from syrag.protocols import Chunker
from syrag.schemas import SourceDocument


class FakeLangChainTextSplitter:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.calls: list[str] = []

    def split_text(self, text: str) -> list[str]:
        self.calls.append(text)
        return self.chunks


class InvalidLangChainTextSplitter:
    pass


@pytest.mark.asyncio
async def test_langchain_text_splitter_chunker_adapts_split_text_output() -> None:
    text_splitter = FakeLangChainTextSplitter(
        chunks=[
            " First chunk. ",
            "Second chunk.",
            "",
        ]
    )
    chunker = LangChainTextSplitterChunker(text_splitter=text_splitter)

    chunks = await chunker.chunk(
        [
            SourceDocument(
                source_id="guide",
                content="SyRAG integrates LangChain text splitters.",
                metadata={"topic": "integrations"},
                page_number=3,
            )
        ]
    )

    assert isinstance(chunker, Chunker)
    assert text_splitter.calls == ["SyRAG integrates LangChain text splitters."]
    assert [chunk.chunk_id for chunk in chunks] == ["guide-chunk-0", "guide-chunk-1"]
    assert [chunk.content for chunk in chunks] == ["First chunk.", "Second chunk."]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert all(chunk.source_id == "guide" for chunk in chunks)
    assert all(chunk.metadata == {"topic": "integrations"} for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks)


def test_langchain_text_splitter_chunker_requires_split_text() -> None:
    with pytest.raises(
        TypeError,
        match=r"text_splitter must expose a callable split_text\(text\) method",
    ):
        LangChainTextSplitterChunker(text_splitter=InvalidLangChainTextSplitter())


@pytest.mark.asyncio
async def test_langchain_text_splitter_chunker_rejects_non_string_chunks() -> None:
    text_splitter = FakeLangChainTextSplitter(chunks=["valid"])
    text_splitter.chunks = ["valid", object()]  # type: ignore[list-item]
    chunker = LangChainTextSplitterChunker(text_splitter=text_splitter)

    with pytest.raises(TypeError, match="must return only strings"):
        await chunker.chunk(
            [
                SourceDocument(
                    source_id="guide",
                    content="SyRAG integrates LangChain.",
                )
            ]
        )
