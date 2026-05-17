# Contributing

Thanks for contributing to SyRAG.

## Development setup

1. Install Python `3.12+`.
2. Install dependencies with `uv sync`.
3. Run checks before opening a change:

```bash
uv run ruff check .
uv run mypy .
uv run pytest
```

Build the documentation site locally when changing files under `docs/`:

```bash
uv run --group docs mkdocs build --strict
```

## Scope

Contributions should preserve these project constraints:

- keep the public framework surface explicit and typed
- avoid adding heavy provider-specific dependencies to the core package
- prefer small, well-tested changes over large speculative rewrites

## Pull requests

- include tests for behavior changes
- update docs when the public API or package behavior changes
- avoid unrelated refactors in the same change
- keep compatibility claims aligned with package metadata, CI, and documented support policy

## Reporting issues

When reporting a bug, include:

- Python version
- installed extras, if any
- minimal reproduction steps
- expected behavior
- actual behavior
