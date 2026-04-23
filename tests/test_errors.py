import pytest

from fastrag.errors import (
    ConfigurationError,
    DependencyConfigurationError,
    FastRAGError,
    PipelineRuntimeError,
    PipelineStageError,
    ProviderError,
    ProviderRequestError,
    ProviderResponseError,
    RateLimitExceededError,
    RequestValidationError,
    SafetyGuardError,
)


@pytest.mark.parametrize(
    ("error", "expected_category", "expected_stage", "expected_status_code"),
    [
        (
            DependencyConfigurationError(
                code="missing_default_component",
                message="Missing embedder default.",
            ),
            "configuration",
            "configuration",
            500,
        ),
        (
            RequestValidationError(
                code="tenant_mismatch",
                message="Tenant mismatch.",
            ),
            "validation",
            "request",
            400,
        ),
        (
            SafetyGuardError(
                code="query_too_large",
                message="Query is too large.",
            ),
            "validation",
            "safety",
            400,
        ),
        (
            RateLimitExceededError(
                code="rate_limited",
                message="Too many requests.",
            ),
            "safety",
            "rate_limit",
            429,
        ),
        (
            ProviderRequestError(
                code="provider_request_failed",
                message="Provider request failed.",
            ),
            "provider",
            "provider",
            500,
        ),
        (
            ProviderResponseError(
                code="provider_invalid_response",
                message="Provider response invalid.",
            ),
            "provider",
            "provider",
            500,
        ),
        (
            PipelineStageError(
                code="embedding_failed",
                message="Embedding failed.",
                stage="embed",
            ),
            "runtime",
            "embed",
            500,
        ),
    ],
)
def test_exception_taxonomy_exposes_expected_defaults(
    error: FastRAGError,
    expected_category: str,
    expected_stage: str,
    expected_status_code: int,
) -> None:
    assert error.error_category == expected_category
    assert error.stage == expected_stage
    assert error.status_code == expected_status_code


def test_exception_taxonomy_preserves_parent_categories() -> None:
    assert issubclass(DependencyConfigurationError, ConfigurationError)
    assert issubclass(SafetyGuardError, RequestValidationError)
    assert issubclass(ProviderRequestError, ProviderError)
    assert issubclass(ProviderResponseError, ProviderError)
    assert issubclass(PipelineStageError, PipelineRuntimeError)
