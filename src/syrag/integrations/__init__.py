__all__: list[str] = []

try:
    from syrag.integrations.langchain import (
        LangChainRetrieverStrategy,
        LangChainTextChunker,
        SyRAGQueryToolInput,
        create_syrag_query_tool,
    )
except ModuleNotFoundError:
    pass
else:
    globals()["SyRAGQueryToolInput"] = SyRAGQueryToolInput
    globals()["create_syrag_query_tool"] = create_syrag_query_tool
    globals()["LangChainRetrieverStrategy"] = LangChainRetrieverStrategy
    globals()["LangChainTextChunker"] = LangChainTextChunker
    __all__.extend(
        [
            "LangChainRetrieverStrategy",
            "LangChainTextChunker",
            "SyRAGQueryToolInput",
            "create_syrag_query_tool",
        ]
    )
