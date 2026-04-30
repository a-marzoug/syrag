# Releasing

SyRAG uses a tag-driven release flow. Publishing should happen from CI, not from a local machine.

## One-Time PyPI Setup

Before the first release, configure a PyPI Trusted Publisher. If the `syrag` project does not exist on PyPI yet, use a pending publisher from your account publishing settings. If it already exists under your account, add the publisher from the project publishing settings.

Use these values:

- PyPI project name: `syrag`
- Owner: `a-marzoug`
- Repository name: `syrag`
- Workflow name: `publish.yml`
- Environment name: leave blank

The workflow publishes when a version tag such as `v0.1.0` is pushed. It does not require a long-lived PyPI API token.

PyPI calls this Trusted Publishing. It uses GitHub Actions OpenID Connect, so the workflow must keep `id-token: write` permission on the publish job.

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

4. Commit the release changes.
5. Make sure the PyPI Trusted Publisher is configured.
6. Create and push a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

7. Confirm the `Publish` GitHub Actions workflow completed successfully.
8. Install the published package in a clean environment and run a smoke test:

```bash
python -m venv /tmp/syrag-smoke
/tmp/syrag-smoke/bin/python -m pip install syrag
/tmp/syrag-smoke/bin/python -c "import importlib.metadata, syrag; print(importlib.metadata.version('syrag')); print(syrag.SyRAG.__name__)"
```

## TestPyPI

TestPyPI automation is not configured yet. For the first public release, use the production PyPI workflow only after the local build and metadata checks pass.

If TestPyPI becomes part of the release process, add a separate workflow or manual dispatch job that publishes to TestPyPI before the production tag is pushed.
