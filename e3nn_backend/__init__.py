"""Runtime backend protocol and registry."""

from .protocol import ArrayBackend
from .registry import get_backend, register_backend, registered_backends

__all__ = ["ArrayBackend", "get_backend", "register_backend", "registered_backends"]
