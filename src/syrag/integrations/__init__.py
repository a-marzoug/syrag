__all__: list[str] = []

try:
    from syrag.integrations.langchain import LangChainTextSplitterChunker
except ModuleNotFoundError:
    pass
else:
    globals().update({"LangChainTextSplitterChunker": LangChainTextSplitterChunker})
    __all__.extend(["LangChainTextSplitterChunker"])
