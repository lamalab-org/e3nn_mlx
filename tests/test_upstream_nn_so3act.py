"""MLX adaptation of upstream ``tests/nn/so3act_test.py``."""

from __future__ import annotations

import pytest

import e3nn_mlx as e3nn
from e3nn_mlx.backend import mlx_backend


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value)))


@pytest.mark.mlx
@pytest.mark.parametrize("lmax", [1, 2, 3, 4])
@pytest.mark.parametrize("act", ["tanh", "square"])
def test_upstream_so3_activation_equivariance(act: str, lmax: int) -> None:
    mx = mlx_backend._require()
    activation = mx.tanh if act == "tanh" else lambda values: values**2
    module = e3nn.SO3Activation(lmax, lmax, activation, 6)
    assert "SO3Activation" in repr(module)
    values = mx.random.normal(shape=(module.irreps_in.dim,))

    for _ in range(10):
        angles = e3nn.rand_angles()
        for inversion in (0, 1):
            d_in = e3nn.irreps_wigner_d(module.irreps_in, *angles, k=inversion)
            d_out = e3nn.irreps_wigner_d(module.irreps_out, *angles, k=inversion)
            actual = module(values @ mx.swapaxes(d_in, -1, -2))
            expected = module(values) @ mx.swapaxes(d_out, -1, -2)
            assert _max_abs(actual - expected) < 0.04


@pytest.mark.mlx
@pytest.mark.parametrize("aspect_ratio", [1, 2, 3, 4])
def test_upstream_so3_activation_identity(aspect_ratio: int) -> None:
    mx = mlx_backend._require()
    module = e3nn.SO3Activation(5, 5, lambda values: values, 6, aspect_ratio=aspect_ratio)
    values = mx.random.normal(shape=(module.irreps_in.dim,))
    expected = module(values)
    compiled = mx.compile(module)
    actual = compiled(values)
    assert _max_abs(actual - expected) < 3e-5
    mse = float(mx.mean((values - expected) ** 2))
    assert mse < 1e-5, mse
