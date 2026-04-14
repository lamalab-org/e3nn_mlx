from __future__ import annotations

import unittest

from e3nn_backend.protocol import ArrayBackend
from e3nn_backend.registry import get_backend, register_backend, registered_backends
from e3nn_mlx.backend import mlx_backend


class DummyBackend(ArrayBackend):
    name = "dummy"

    def is_available(self) -> bool:
        return True

    def asarray(self, value, /, *, dtype=None):
        return value

    def zeros(self, shape, /, *, dtype=None):
        return [[0] * shape[-1]]

    def reshape(self, array, shape, /):
        return array

    def concatenate(self, arrays, /, *, axis=0):
        return list(arrays)

    def einsum(self, subscripts, *operands):
        return (subscripts, operands)

    def compile(self, fn, /):
        return fn


class BackendRegistryTest(unittest.TestCase):
    def test_register_backend(self) -> None:
        backend = register_backend(DummyBackend())
        self.assertIs(backend, get_backend("dummy"))
        self.assertIn("dummy", registered_backends())

    def test_mlx_backend_is_registered(self) -> None:
        self.assertIs(get_backend("mlx"), mlx_backend)
        self.assertIsInstance(mlx_backend.is_available(), bool)


if __name__ == "__main__":
    unittest.main()
