"""Hooks that let an array runtime provide backend-dependent irrep operations.

``e3nn_core`` is deliberately free of any array dependency, but upstream e3nn
exposes several methods on ``Irrep``/``Irreps`` that must return arrays
(``D_from_angles`` and friends, ``randn``).  Those methods are declared here as
thin dispatchers; importing :mod:`e3nn_mlx` installs the MLX implementations.
"""

from __future__ import annotations

from typing import Any, Callable

_HOOKS: dict[str, Callable[..., Any]] = {}


def register_runtime(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
    """Install the implementation of a backend-dependent operation."""

    if not name:
        raise ValueError("runtime hook name must be non-empty")
    _HOOKS[name] = function
    return function


def get_runtime(name: str) -> Callable[..., Any]:
    try:
        return _HOOKS[name]
    except KeyError:
        raise RuntimeError(
            f"{name!r} needs an array runtime, which e3nn_core does not provide; "
            "import e3nn_mlx to install the MLX implementation"
        ) from None


def registered_runtimes() -> tuple[str, ...]:
    return tuple(sorted(_HOOKS))
