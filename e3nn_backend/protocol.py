"""Thin backend protocol for numerical backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ArrayBackend(Protocol):
    name: str

    def is_available(self) -> bool:
        """Return whether the backend runtime is importable and usable."""

    def asarray(self, value: Any, /, *, dtype: Any | None = None) -> Any:
        """Convert a Python value into a backend array."""

    def zeros(self, shape: tuple[int, ...], /, *, dtype: Any | None = None) -> Any:
        """Create a zero-filled array."""

    def reshape(self, array: Any, shape: tuple[int, ...], /) -> Any:
        """Reshape an array."""

    def concatenate(self, arrays: tuple[Any, ...], /, *, axis: int = 0) -> Any:
        """Concatenate arrays."""

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        """Execute an einsum expression."""

    def compile(self, fn: Any, /) -> Any:
        """Optionally stage or compile a callable."""
