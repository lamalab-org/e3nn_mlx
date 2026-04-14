"""Basic MLX tensor helpers and extension seams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .compat import require_mlx


@dataclass(frozen=True, slots=True)
class ExtensionKernel:
    name: str
    fn: Callable[..., Any]
    has_custom_vjp: bool = False
    has_custom_jvp: bool = False
    note: str = ""


_EXTENSIONS: dict[str, ExtensionKernel] = {}


def register_extension(kernel: ExtensionKernel) -> ExtensionKernel:
    _EXTENSIONS[kernel.name] = kernel
    return kernel


def get_extension(name: str) -> ExtensionKernel | None:
    return _EXTENSIONS.get(name)


def registered_extensions() -> tuple[str, ...]:
    return tuple(sorted(_EXTENSIONS))


def compile_or_identity(fn: Callable[..., Any], *, enabled: bool = True) -> Callable[..., Any]:
    if not enabled:
        return fn
    mx, _ = require_mlx()
    return mx.compile(fn)


def eval_maybe(value: Any) -> Any:
    mx, _ = require_mlx()
    if hasattr(value, "array"):
        mx.eval(value.array)
    elif isinstance(value, tuple):
        for item in value:
            eval_maybe(item)
    else:
        try:
            mx.eval(value)
        except TypeError:
            pass
    return value
