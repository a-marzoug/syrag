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
    from syrag.providers.faiss import FAISSVectorStore
except ModuleNotFoundError:
    pass
else:
    globals().update({"FAISSVectorStore": FAISSVectorStore})
    __all__.extend(["FAISSVectorStore"])

try:
    from syrag.providers.qdrant import QdrantVectorStore
except ModuleNotFoundError:
    pass
else:
    globals().update({"QdrantVectorStore": QdrantVectorStore})
    __all__.extend(["QdrantVectorStore"])

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

try:
    from syrag.providers.google import GoogleEmbedder, GoogleLLM
except ModuleNotFoundError:
    pass
else:
    globals().update(
        {
            "GoogleEmbedder": GoogleEmbedder,
            "GoogleLLM": GoogleLLM,
        }
    )
    __all__.extend(["GoogleEmbedder", "GoogleLLM"])
