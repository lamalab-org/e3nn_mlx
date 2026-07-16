"""Small adapters shared by the e3nn-style public namespaces."""

from __future__ import annotations

from typing import Any

from .irreps_array import IrrepsArray


def ensure_irreps_array(value: Any, irreps) -> tuple[IrrepsArray, bool]:
    """Return a typed value and whether the caller supplied a raw MLX array."""

    if isinstance(value, IrrepsArray):
        return value, False
    return IrrepsArray(irreps, value), True


def unwrap_irreps_arrays(value: Any, raw: bool) -> Any:
    """Restore upstream-style raw outputs without disturbing typed callers."""

    if not raw:
        return value
    if isinstance(value, IrrepsArray):
        return value.array
    if isinstance(value, tuple):
        return tuple(unwrap_irreps_arrays(item, True) for item in value)
    if isinstance(value, list):
        return [unwrap_irreps_arrays(item, True) for item in value]
    return value
