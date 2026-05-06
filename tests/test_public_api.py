import syrag

PRIMARY_APPLICATION_EXPORTS = {
    "SyRAG",
    "create_app",
    "Settings",
    "IngestRequest",
    "IngestResponse",
    "QueryRequest",
    "RAGResponse",
    "Citation",
    "SourceDocument",
    "DocumentChunk",
    "RetrievedChunk",
    "RequestContext",
    "__version__",
}

EXTENSION_PROTOCOL_EXPORTS = {
    "Chunker",
    "Embedder",
    "VectorStore",
    "RetrievalStrategy",
    "PromptAssembler",
    "GenerationPolicy",
    "LLM",
    "RequestContextHook",
    "AuthHook",
    "RateLimiter",
    "SafetyGuard",
}

FIRST_PARTY_COMPONENT_EXPORTS = {
    "PassThroughChunker",
    "InMemoryEmbedder",
    "InMemoryVectorStore",
    "InMemoryLLM",
    "SQLiteVectorStore",
    "DefaultRequestContextHook",
    "NoOpAuthHook",
    "DefaultSafetyGuard",
    "InMemoryRateLimiter",
    "StructuredLogging",
    "JSONLogFormatter",
    "OpenTelemetryTracing",
}


def test_primary_public_api_exports_are_available() -> None:
    exports = set(syrag.__all__)

    assert PRIMARY_APPLICATION_EXPORTS <= exports
    assert EXTENSION_PROTOCOL_EXPORTS <= exports
    assert FIRST_PARTY_COMPONENT_EXPORTS <= exports


def test_public_api_exports_are_unique_and_resolvable() -> None:
    assert len(syrag.__all__) == len(set(syrag.__all__))

    for export_name in syrag.__all__:
        assert hasattr(syrag, export_name), export_name


def test_optional_provider_exports_are_available_in_dev_environment() -> None:
    exports = set(syrag.__all__)

    assert "ChromaVectorStore" in exports
    assert "FAISSVectorStore" in exports
    assert "GoogleEmbedder" in exports
    assert "GoogleLLM" in exports
    assert "LangChainTextChunker" in exports
    assert "OpenAIEmbedder" in exports
    assert "OpenAILLM" in exports
