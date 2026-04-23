from fastrag.app import FastRAG, app, create_app
from fastrag.bootstrap import BootstrapService
from fastrag.config import (
    BootstrapSettings,
    ComponentDefaults,
    InMemoryProviderSettings,
    ProviderSettings,
    Settings,
    get_settings,
)
from fastrag.dependencies import ComponentResolver
from fastrag.errors import (
    ConfigurationError,
    DependencyConfigurationError,
    FastRAGError,
    PipelineRuntimeError,
    PipelineStageError,
    ProviderError,
    ProviderRequestError,
    ProviderResponseError,
    RateLimitExceededError,
    RequestValidationError,
    SafetyGuardError,
)
from fastrag.guardrails import DefaultSafetyGuard, InMemoryRateLimiter
from fastrag.hooks import DefaultRequestContextHook, NoOpAuthHook
from fastrag.protocols import (
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
from fastrag.providers import (
    InMemoryEmbedder,
    InMemoryLLM,
    InMemoryProviderFactory,
    InMemoryVectorStore,
    PassThroughChunker,
    ProviderFactory,
    SQLiteVectorStore,
)
from fastrag.registry import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
    ComponentRegistry,
    ComponentValidationError,
    RegistryError,
)
from fastrag.schemas import (
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
from fastrag.services import (
    DefaultGenerationPolicy,
    DefaultPromptAssembler,
    DefaultRetrievalStrategy,
    RetrievalStrategy,
)
from fastrag.structured_logging import JSONLogFormatter, StructuredLogging
from fastrag.tracing import OpenTelemetryTracing

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
    "FastRAGError",
    "FastRAG",
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
    "app",
    "create_app",
    "get_settings",
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

try:
    from fastrag.testing import (
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
