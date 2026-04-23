from __future__ import annotations

from typing import Any, ClassVar


class FastRAGError(Exception):
    """Base framework exception with HTTP and stage metadata."""

    category: ClassVar[str] = "runtime"
    default_stage: ClassVar[str] = "runtime"
    default_status_code: ClassVar[int] = 500

    def __init__(
        self,
        *,
        code: str,
        message: str,
        stage: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage or self.default_stage
        self.status_code = (
            self.default_status_code if status_code is None else status_code
        )
        self.details = details or {}
        self.error_category = type(self).category


class ConfigurationError(FastRAGError):
    """Raised when framework configuration is incomplete or inconsistent."""

    category = "configuration"
    default_stage = "configuration"


class DependencyConfigurationError(ConfigurationError):
    """Raised when dependency resolution cannot satisfy a required component."""


class RequestValidationError(FastRAGError):
    """Raised when a request is invalid before pipeline execution begins."""

    category = "validation"
    default_stage = "request"
    default_status_code = 400


class SafetyGuardError(RequestValidationError):
    """Raised when a request exceeds configured safety limits."""

    default_stage = "safety"


class RateLimitExceededError(FastRAGError):
    """Raised when a request exceeds configured throughput limits."""

    category = "safety"
    default_stage = "rate_limit"
    default_status_code = 429


class ProviderError(FastRAGError):
    """Raised when an external provider fails or returns invalid data."""

    category = "provider"
    default_stage = "provider"


class ProviderRequestError(ProviderError):
    """Raised when a provider HTTP/API request fails."""


class ProviderResponseError(ProviderError):
    """Raised when a provider returns an invalid or incomplete response."""


class PipelineRuntimeError(FastRAGError):
    """Raised when a runtime pipeline stage fails during execution."""

    category = "runtime"


class PipelineStageError(PipelineRuntimeError):
    """Raised when a pipeline stage fails during execution."""
