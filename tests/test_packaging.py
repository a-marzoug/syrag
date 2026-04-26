import tomllib
from pathlib import Path

import fastrag
from fastrag._optional import missing_optional_dependency
from fastrag.providers import __all__ as provider_exports


def test_pyproject_declares_optional_extension_boundaries() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    dependencies = project["dependencies"]
    optional_dependencies = project["optional-dependencies"]

    assert "httpx>=0.28.1" not in dependencies
    assert "uvicorn[standard]>=0.44.0" not in dependencies
    assert optional_dependencies["openai"] == ["httpx>=0.28.1"]
    assert optional_dependencies["testing"] == ["httpx>=0.28.1"]
    assert optional_dependencies["server"] == ["uvicorn[standard]>=0.44.0"]


def test_pyproject_declares_release_metadata() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["license"] == "Apache-2.0"
    assert project["authors"] == [{"name": "A. Marzoug"}]
    assert project["requires-python"] == ">=3.12,<3.14"
    assert project["urls"] == {
        "Repository": "https://github.com/a-marzoug/fastrag",
        "Issues": "https://github.com/a-marzoug/fastrag/issues",
        "Documentation": "https://github.com/a-marzoug/fastrag/tree/main/docs",
    }
    assert "Programming Language :: Python :: 3.12" in project["classifiers"]
    assert "Programming Language :: Python :: 3.13" in project["classifiers"]


def test_optional_dependency_error_mentions_install_extra() -> None:
    error = missing_optional_dependency(feature="fastrag.testing", extra="testing")

    assert str(error) == (
        "fastrag.testing requires the optional 'testing' extra. "
        "Install with `pip install fastrag[testing]`."
    )


def test_optional_integrations_are_exported_when_installed() -> None:
    assert "OpenAIEmbedder" in provider_exports
    assert "OpenAIEmbedder" in fastrag.__all__
    assert "create_test_client" in fastrag.__all__
