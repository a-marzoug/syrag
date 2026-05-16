import tomllib
from pathlib import Path

import syrag
from syrag._optional import missing_optional_dependency
from syrag.providers import __all__ as provider_exports


def test_pyproject_declares_optional_extension_boundaries() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    dependencies = project["dependencies"]
    optional_dependencies = project["optional-dependencies"]

    assert project["name"] == "syrag"
    assert "httpx>=0.28.1" not in dependencies
    assert "uvicorn[standard]>=0.44.0" not in dependencies
    assert optional_dependencies["chroma"] == ["chromadb>=1.0.0"]
    assert optional_dependencies["faiss"] == ["faiss-cpu>=1.8.0"]
    assert optional_dependencies["google"] == ["google-genai>=1.0.0"]
    assert optional_dependencies["langchain"] == [
        "httpx>=0.28.1",
        "langchain-core>=1.0.0",
        "langchain-text-splitters>=0.3.0",
    ]
    assert optional_dependencies["llamaindex"] == [
        "httpx>=0.28.1",
        "llama-index-core>=0.14.0",
    ]
    assert optional_dependencies["openai"] == ["httpx>=0.28.1"]
    assert optional_dependencies["testing"] == ["httpx>=0.28.1"]
    assert optional_dependencies["server"] == ["uvicorn[standard]>=0.44.0"]


def test_optional_extras_are_documented_in_compatibility_matrix() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional_dependencies = pyproject["project"]["optional-dependencies"]
    compatibility_doc = Path("docs/compatibility.md").read_text(encoding="utf-8")

    for extra_name, dependencies in optional_dependencies.items():
        assert f"`{extra_name}`" in compatibility_doc
        assert f'syrag[{extra_name}]' in compatibility_doc
        for dependency in dependencies:
            package_name = dependency.split(">=", maxsplit=1)[0]
            assert package_name in compatibility_doc


def test_pyproject_declares_release_metadata() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert syrag.__version__ == project["version"]
    assert project["license"] == "Apache-2.0"
    assert project["authors"] == [{"name": "A. Marzoug"}]
    assert project["requires-python"] == ">=3.12,<3.14"
    assert project["urls"] == {
        "Repository": "https://github.com/a-marzoug/syrag",
        "Issues": "https://github.com/a-marzoug/syrag/issues",
        "Documentation": "https://a-marzoug.github.io/syrag/",
    }
    assert "Programming Language :: Python :: 3.12" in project["classifiers"]
    assert "Programming Language :: Python :: 3.13" in project["classifiers"]


def test_optional_dependency_error_mentions_install_extra() -> None:
    error = missing_optional_dependency(feature="syrag.testing", extra="testing")

    assert str(error) == (
        "syrag.testing requires the optional 'testing' extra. "
        "Install with `pip install syrag[testing]`."
    )


def test_optional_integrations_are_exported_when_installed() -> None:
    assert "__version__" in syrag.__all__
    assert "ChromaVectorStore" in provider_exports
    assert "ChromaVectorStore" in syrag.__all__
    assert "FAISSVectorStore" in provider_exports
    assert "FAISSVectorStore" in syrag.__all__
    assert "GoogleEmbedder" in provider_exports
    assert "GoogleEmbedder" in syrag.__all__
    assert "GoogleLLM" in provider_exports
    assert "GoogleLLM" in syrag.__all__
    assert "LangChainRetrieverStrategy" in syrag.__all__
    assert "LangChainTextChunker" in syrag.__all__
    assert "LlamaIndexNodeChunker" in syrag.__all__
    assert "LlamaIndexRetrieverStrategy" in syrag.__all__
    assert "SyRAGQueryEngine" in syrag.__all__
    assert "SyRAGQueryToolInput" in syrag.__all__
    assert "create_syrag_query_tool" in syrag.__all__
    assert "OpenAIEmbedder" in provider_exports
    assert "OpenAIEmbedder" in syrag.__all__
    assert "create_test_client" in syrag.__all__
