from __future__ import annotations

import pytest

from e3nn_mlx.compat import mlx_runtime_available


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "mlx: requires a working MLX runtime")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "mlx" in item.keywords and not mlx_runtime_available():
        pytest.skip("MLX runtime is not available in this execution context")
