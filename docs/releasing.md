# Releasing

SyRAG uses a tag-driven release flow. Publishing should happen from CI, not from a local machine.

## One-Time PyPI Setup

Before the first release, configure PyPI Trusted Publishing for:

- PyPI project: `syrag`
- GitHub owner/repository: `a-marzoug/syrag`
- Workflow file: `.github/workflows/publish.yml`

The workflow publishes when a version tag such as `v0.1.0` is pushed. It does not require a long-lived PyPI API token.

## Release Checklist

1. Confirm `pyproject.toml` has the intended version.
2. Review `CHANGELOG.md` and replace `Unreleased` with the release date.
3. Run the local checks:

```bash
uv run ruff check .
uv run mypy .
uv run pytest
uv build
uvx twine check dist/*
```

1. Commit the release changes.
2. Create and push a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

1. Confirm the `Publish` GitHub Actions workflow completed successfully.
2. Install the published package in a clean environment and run a smoke test:

```bash
python -m venv /tmp/syrag-smoke
/tmp/syrag-smoke/bin/python -m pip install syrag
/tmp/syrag-smoke/bin/python -c "import syrag; print(syrag.__version__)"
```

## TestPyPI

TestPyPI automation is not configured yet. For the first public release, use the production PyPI workflow only after the local build and metadata checks pass.

If TestPyPI becomes part of the release process, add a separate workflow or manual dispatch job that publishes to TestPyPI before the production tag is pushed.
