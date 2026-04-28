# FastRAG Docs

FastRAG is a production-first Python framework for building Retrieval-Augmented Generation (RAG) services with a FastAPI-like developer experience.

This documentation set describes the framework that exists in this repository today. It focuses on the public API, current architectural boundaries, and shipped extension points.

## Reading Guide

- [Product Overview](./overview.md): What FastRAG ships, who it serves, and where it draws the line.
- [Architecture](./architecture.md): Current runtime structure, request flow, and packaging boundaries.
- [Component Contracts](./component-contracts.md): The protocol-first extension points that applications and providers rely on.
- [MVP Status](./mvp-roadmap.md): What is implemented now and what remains intentionally out of scope.

## Current Status

The framework is past bootstrap. The current codebase includes:

- typed ingest and query routes
- pluggable providers and pipeline stages
- request context, auth hooks, tenant scoping, guardrails, and rate limiting
- OpenAPI examples, tracing hooks, and structured logging
- a testing toolkit and provider contract coverage

## Documentation Principles

- Documentation tracks implementation reality, not aspirational design.
- Public contracts are described ahead of provider-specific details.
- Operational concerns such as observability, security, and multi-tenancy remain first-class.
- Optional extras are called out explicitly so the core package boundary stays clear.
