"""MLX backend skeleton."""

from __future__ import annotations

from typing import Any

from e3nn_backend.protocol import ArrayBackend
from e3nn_backend.registry import register_backend
from .compat import mlx_runtime_available


class MLXBackend(ArrayBackend):
    name = "mlx"

    def __init__(self) -> None:
        self._mx = None

    def is_available(self) -> bool:
        return mlx_runtime_available()

    def _require(self) -> Any:
        if self._mx is None:
            import mlx.core as mx  # type: ignore[import-not-found]

            self._mx = mx
        return self._mx

    def asarray(self, value: Any, /, *, dtype: Any | None = None) -> Any:
        mx = self._require()
        return mx.array(value, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], /, *, dtype: Any | None = None) -> Any:
        mx = self._require()
        return mx.zeros(shape, dtype=dtype)

    def reshape(self, array: Any, shape: tuple[int, ...], /) -> Any:
        mx = self._require()
        return mx.reshape(array, shape)

    def concatenate(self, arrays: tuple[Any, ...], /, *, axis: int = 0) -> Any:
        mx = self._require()
        return mx.concatenate(arrays, axis=axis)

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        mx = self._require()
        return mx.einsum(subscripts, *operands)

    def compile(self, fn: Any, /) -> Any:
        mx = self._require()
        return mx.compile(fn)


mlx_backend = register_backend(MLXBackend())
