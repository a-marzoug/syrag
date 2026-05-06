__all__: list[str] = []

try:
    from syrag.integrations.langchain import LangChainTextChunker
except ModuleNotFoundError:
    pass
else:
    globals().update({"LangChainTextChunker": LangChainTextChunker})
    __all__.extend(["LangChainTextChunker"])
