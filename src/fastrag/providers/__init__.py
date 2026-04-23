from fastrag.providers.chunking import PassThroughChunker
from fastrag.providers.factories import InMemoryProviderFactory, ProviderFactory
from fastrag.providers.in_memory import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore
from fastrag.providers.sqlite import SQLiteVectorStore

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
    from fastrag.providers.openai import OpenAIEmbedder, OpenAILLM
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
