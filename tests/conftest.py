from __future__ import annotations

import os
import zlib

import numpy as np
import pytest

from e3nn_mlx.compat import mlx_runtime_available


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "mlx: requires a working MLX runtime")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "mlx" in item.keywords and not mlx_runtime_available():
        if os.environ.get("E3NN_MLX_REQUIRE_RUNTIME") == "1":
            pytest.fail("MLX runtime is required in this test job but is unavailable", pytrace=False)
        pytest.skip("MLX runtime is not available in this execution context")


@pytest.fixture(autouse=True)
def _deterministic_rng(request: pytest.FixtureRequest) -> None:
    """Seed every global RNG per test so runs are reproducible.

    Most tests draw random inputs and then assert an absolute tolerance, so an
    unseeded draw makes them pass or fail by luck and makes a CI failure
    impossible to reproduce locally. Deriving the seed from the test id keeps
    each test deterministic while still giving different tests different data.
    Tests that seed explicitly still override this. Set ``E3NN_MLX_TEST_SEED``
    to sweep a different draw for every test at once, so a scheduled CI job can
    explore the random space that a fixed seed would otherwise pin shut.
    """

    salt = os.environ.get("E3NN_MLX_TEST_SEED", "")
    seed = zlib.crc32(f"{salt}:{request.node.nodeid}".encode()) & 0x7FFFFFFF
    np.random.seed(seed)
    if mlx_runtime_available():
        import mlx.core as mx

        mx.random.seed(seed)
