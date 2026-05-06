from importlib.metadata import PackageNotFoundError, version

from syrag.app import SyRAG, app, create_app
from syrag.bootstrap import BootstrapService
from syrag.config import (
    BootstrapSettings,
    ComponentDefaults,
    InMemoryProviderSettings,
    ProviderSettings,
    Settings,
    get_settings,
)
from syrag.dependencies import ComponentResolver
from syrag.errors import (
    ConfigurationError,
    DependencyConfigurationError,
    PipelineRuntimeError,
    PipelineStageError,
    ProviderError,
    ProviderRequestError,
    ProviderResponseError,
    RateLimitExceededError,
    RequestValidationError,
    SafetyGuardError,
    SyRAGError,
)
from syrag.guardrails import DefaultSafetyGuard, InMemoryRateLimiter
from syrag.hooks import DefaultRequestContextHook, NoOpAuthHook
from syrag.protocols import (
    LLM,
    AuthHook,
    Chunker,
    Embedder,
    GenerationPolicy,
    PromptAssembler,
    RateLimiter,
    RequestContextHook,
    SafetyGuard,
    VectorStore,
)
from syrag.providers import (
    InMemoryEmbedder,
    InMemoryLLM,
    InMemoryProviderFactory,
    InMemoryVectorStore,
    PassThroughChunker,
    ProviderFactory,
    SQLiteVectorStore,
)
from syrag.registry import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
    ComponentRegistry,
    ComponentValidationError,
    RegistryError,
)
from syrag.schemas import (
    AssembledPrompt,
    Citation,
    DocumentChunk,
    GenerationRequest,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    RAGResponse,
    RequestContext,
    RetrievedChunk,
    RetrievedDocument,
    SourceDocument,
)
from syrag.services import (
    DefaultGenerationPolicy,
    DefaultPromptAssembler,
    DefaultRetrievalStrategy,
    RetrievalStrategy,
)
from syrag.structured_logging import JSONLogFormatter, StructuredLogging
from syrag.tracing import OpenTelemetryTracing

try:
    __version__ = version("syrag")
except PackageNotFoundError:  # pragma: no cover - only occurs from an unpackaged source tree
    __version__ = "0.1.0"

__all__ = [
    "ComponentAlreadyRegisteredError",
    "ComponentNotFoundError",
    "ComponentRegistry",
    "ComponentValidationError",
    "BootstrapService",
    "BootstrapSettings",
    "ComponentDefaults",
    "ComponentResolver",
    "ConfigurationError",
    "AssembledPrompt",
    "AuthHook",
    "Citation",
    "Chunker",
    "DefaultRequestContextHook",
    "DefaultSafetyGuard",
    "DefaultGenerationPolicy",
    "DefaultPromptAssembler",
    "DependencyConfigurationError",
    "DocumentChunk",
    "DefaultRetrievalStrategy",
    "Embedder",
    "SyRAGError",
    "SyRAG",
    "GenerationPolicy",
    "GenerationRequest",
    "InMemoryProviderSettings",
    "IngestRequest",
    "IngestResponse",
    "InMemoryEmbedder",
    "InMemoryLLM",
    "InMemoryProviderFactory",
    "InMemoryVectorStore",
    "JSONLogFormatter",
    "LLM",
    "OpenTelemetryTracing",
    "NoOpAuthHook",
    "InMemoryRateLimiter",
    "PassThroughChunker",
    "PromptAssembler",
    "ProviderSettings",
    "ProviderFactory",
    "ProviderError",
    "ProviderRequestError",
    "ProviderResponseError",
    "PipelineRuntimeError",
    "PipelineStageError",
    "QueryRequest",
    "RAGResponse",
    "RateLimiter",
    "RateLimitExceededError",
    "RequestContext",
    "RequestContextHook",
    "RequestValidationError",
    "RegistryError",
    "RetrievalStrategy",
    "RetrievedChunk",
    "RetrievedDocument",
    "SafetyGuardError",
    "SourceDocument",
    "Settings",
    "SQLiteVectorStore",
    "StructuredLogging",
    "SafetyGuard",
    "VectorStore",
    "__version__",
    "app",
    "create_app",
    "get_settings",
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

try:
    from syrag.integrations.langchain import LangChainRetrieverStrategy, LangChainTextChunker
except ModuleNotFoundError:
    pass
else:
    globals()["LangChainRetrieverStrategy"] = LangChainRetrieverStrategy
    globals()["LangChainTextChunker"] = LangChainTextChunker
    __all__.extend(["LangChainRetrieverStrategy", "LangChainTextChunker"])

try:
    from syrag.testing import (
        EmbedCall,
        FakeChunker,
        FakeEmbedder,
        FakeLLM,
        FakeProviderBundle,
        FakeVectorStore,
        GenerateCall,
        QueryCall,
        UpsertCall,
        create_test_app,
        create_test_client,
        seed_documents,
    )
except ModuleNotFoundError:
    pass
else:
    globals().update(
        {
            "EmbedCall": EmbedCall,
            "FakeChunker": FakeChunker,
            "FakeEmbedder": FakeEmbedder,
            "FakeLLM": FakeLLM,
            "FakeProviderBundle": FakeProviderBundle,
            "FakeVectorStore": FakeVectorStore,
            "GenerateCall": GenerateCall,
            "QueryCall": QueryCall,
            "UpsertCall": UpsertCall,
            "create_test_app": create_test_app,
            "create_test_client": create_test_client,
            "seed_documents": seed_documents,
        }
    )
    __all__.extend(
        [
            "EmbedCall",
            "FakeChunker",
            "FakeEmbedder",
            "FakeLLM",
            "FakeProviderBundle",
            "FakeVectorStore",
            "GenerateCall",
            "QueryCall",
            "UpsertCall",
            "create_test_app",
            "create_test_client",
            "seed_documents",
        ]
    )
