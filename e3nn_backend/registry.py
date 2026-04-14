"""Runtime backend registry."""

from __future__ import annotations

from typing import Dict

from .protocol import ArrayBackend

_REGISTRY: Dict[str, ArrayBackend] = {}


def register_backend(backend: ArrayBackend) -> ArrayBackend:
    if not backend.name:
        raise ValueError("backend.name must be non-empty")
    _REGISTRY[backend.name] = backend
    return backend


def get_backend(name: str) -> ArrayBackend:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown backend {name!r}; available backends: {available}") from exc


def registered_backends() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
