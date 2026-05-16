# Documentation Site

SyRAG uses MkDocs Material for the hosted documentation site. This is the same documentation toolchain style used by FastAPI: Markdown source files live in `docs/`, `mkdocs.yml` controls navigation and theme behavior, and GitHub Pages serves the generated static site.

## Local Preview

Install the docs dependency group and start the preview server:

```bash
uv run --group docs mkdocs serve
```

MkDocs will print a local URL, usually `http://127.0.0.1:8000/`.

## Strict Build

Run the same strict build used by CI:

```bash
uv run --group docs mkdocs build --strict
```

Strict mode fails on broken internal links and invalid navigation entries. Use it before pushing documentation changes.

## Publishing

The docs workflow in `.github/workflows/docs.yml` does two things:

- Pull requests build the docs with `mkdocs build --strict`.
- Pushes to `main` build and deploy the site to GitHub Pages.

To enable the hosted site in GitHub:

1. Open the repository settings.
2. Go to **Pages**.
3. Set **Build and deployment** to **GitHub Actions**.
4. Push a docs change to `main`.

After the workflow succeeds, the site should be available at:

```text
https://a-marzoug.github.io/syrag/
```

## Writing Guidelines

- Keep docs aligned with released behavior.
- Prefer complete runnable examples over fragments when documenting integrations.
- Put provider-specific setup in cookbook or provider pages, not in the core overview.
- Keep optional dependency requirements explicit.
- Use admonitions for operational caveats, security notes, and production warnings.
