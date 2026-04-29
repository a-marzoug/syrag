from syrag.providers.chunking import PassThroughChunker
from syrag.providers.factories import InMemoryProviderFactory, ProviderFactory
from syrag.providers.in_memory import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from syrag.providers.sqlite import SQLiteVectorStore

__all__ = [
    "InMemoryEmbedder",
    "InMemoryLLM",
    "InMemoryProviderFactory",
    "InMemoryVectorStore",
    "PassThroughChunker",
    "ProviderFactory",
    "SQLiteVectorStore",
]

try:
    from syrag.providers.chroma import ChromaVectorStore
except ModuleNotFoundError:
    pass
else:
    globals().update({"ChromaVectorStore": ChromaVectorStore})
    __all__.extend(["ChromaVectorStore"])

try:
    from syrag.providers.openai import OpenAIEmbedder, OpenAILLM
except ModuleNotFoundError:
    pass
else:
    globals().update(
        {
            "OpenAIEmbedder": OpenAIEmbedder,
            "OpenAILLM": OpenAILLM,
        }
    )
    __all__.extend(["OpenAIEmbedder", "OpenAILLM"])
