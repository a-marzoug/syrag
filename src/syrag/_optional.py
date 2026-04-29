from __future__ import annotations


def missing_optional_dependency(*, feature: str, extra: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(
        f"{feature} requires the optional '{extra}' extra. "
        f"Install with `pip install syrag[{extra}]`."
    )
