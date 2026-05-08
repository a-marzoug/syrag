from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import venv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SmokeCase:
    name: str
    install_target: str
    code: str


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    wheel = _resolve_wheel(args.wheel)
    cases = _smoke_cases(wheel)
    selected_cases = [
        smoke_case for smoke_case in cases if not args.case or smoke_case.name in args.case
    ]

    if args.case and len(selected_cases) != len(args.case):
        available_cases = {smoke_case.name for smoke_case in cases}
        unknown_cases = sorted(set(args.case) - available_cases)
        raise SystemExit(f"Unknown smoke case(s): {', '.join(unknown_cases)}")

    with tempfile.TemporaryDirectory(prefix="syrag-smoke-") as temp_dir:
        workspace = Path(temp_dir)
        for smoke_case in selected_cases:
            _run_smoke_case(smoke_case, workspace=workspace)

    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the built SyRAG wheel in clean venvs and smoke-test imports.",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="Path to the wheel to test. Defaults to the single wheel in dist/.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[
            "core",
            "openai",
            "google",
            "chroma",
            "faiss",
            "langchain",
            "llamaindex",
        ],
        help="Run one smoke case. May be passed multiple times.",
    )
    return parser.parse_args(argv)


def _resolve_wheel(wheel: Path | None) -> Path:
    if wheel is not None:
        if not wheel.is_file():
            raise SystemExit(f"Wheel does not exist: {wheel}")
        return wheel

    wheels = sorted(Path("dist").glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(
            f"Expected exactly one wheel in dist/, found {len(wheels)}. "
            "Run `uv build` first or pass --wheel."
        )
    return wheels[0]


def _smoke_cases(wheel: Path) -> list[SmokeCase]:
    wheel_path = wheel.resolve()
    return [
        SmokeCase(
            name="core",
            install_target=str(wheel_path),
            code="""
import importlib.metadata as metadata

import syrag

assert metadata.version("syrag") == syrag.__version__
assert syrag.SyRAG.__name__ == "SyRAG"
assert "OpenAIEmbedder" not in syrag.__all__
assert "ChromaVectorStore" not in syrag.__all__
""",
        ),
        SmokeCase(
            name="openai",
            install_target=f"{wheel_path}[openai]",
            code="""
from syrag import OpenAIEmbedder, OpenAILLM

assert OpenAIEmbedder.__name__ == "OpenAIEmbedder"
assert OpenAILLM.__name__ == "OpenAILLM"
""",
        ),
        SmokeCase(
            name="google",
            install_target=f"{wheel_path}[google]",
            code="""
from syrag import GoogleEmbedder, GoogleLLM

assert GoogleEmbedder.__name__ == "GoogleEmbedder"
assert GoogleLLM.__name__ == "GoogleLLM"
""",
        ),
        SmokeCase(
            name="chroma",
            install_target=f"{wheel_path}[chroma]",
            code="""
from syrag import ChromaVectorStore

assert ChromaVectorStore.__name__ == "ChromaVectorStore"
""",
        ),
        SmokeCase(
            name="faiss",
            install_target=f"{wheel_path}[faiss]",
            code="""
from syrag import FAISSVectorStore

assert FAISSVectorStore.__name__ == "FAISSVectorStore"
""",
        ),
        SmokeCase(
            name="langchain",
            install_target=f"{wheel_path}[langchain]",
            code="""
from syrag import (
    LangChainRetrieverStrategy,
    LangChainTextChunker,
    SyRAGQueryToolInput,
    create_syrag_query_tool,
)

assert LangChainRetrieverStrategy.__name__ == "LangChainRetrieverStrategy"
assert LangChainTextChunker.__name__ == "LangChainTextChunker"
assert SyRAGQueryToolInput.__name__ == "SyRAGQueryToolInput"
assert callable(create_syrag_query_tool)
""",
        ),
        SmokeCase(
            name="llamaindex",
            install_target=f"{wheel_path}[llamaindex]",
            code="""
from syrag import (
    LlamaIndexNodeChunker,
    LlamaIndexRetrieverStrategy,
    SyRAGQueryEngine,
)

assert LlamaIndexNodeChunker.__name__ == "LlamaIndexNodeChunker"
assert LlamaIndexRetrieverStrategy.__name__ == "LlamaIndexRetrieverStrategy"
assert SyRAGQueryEngine.__name__ == "SyRAGQueryEngine"
""",
        ),
    ]


def _run_smoke_case(smoke_case: SmokeCase, *, workspace: Path) -> None:
    venv_dir = workspace / smoke_case.name
    print(f"::group::package smoke: {smoke_case.name}", flush=True)
    try:
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python = _python_executable(venv_dir)
        _run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                smoke_case.install_target,
            ]
        )
        _run([python, "-c", smoke_case.code])
    finally:
        print("::endgroup::", flush=True)


def _python_executable(venv_dir: Path) -> str:
    if sys.platform == "win32":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def _run(command: Iterable[str]) -> None:
    subprocess.run(list(command), check=True)


if __name__ == "__main__":
    raise SystemExit(main())
