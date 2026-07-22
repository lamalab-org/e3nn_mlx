"""MLX adaptation of upstream ``tests/nn/s2act_test.py``."""

from __future__ import annotations

import itertools

import pytest

import e3nn_mlx as e3nn
from e3nn_core.irreps import Irrep, Irreps, MulIrrep
from e3nn_mlx.backend import mlx_backend


def _spherical_tensor(lmax: int, p_val: int, p_arg: int) -> Irreps:
    return Irreps([MulIrrep(1, Irrep(l, p_val * p_arg**l)) for l in range(lmax + 1)])


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value)))


@pytest.mark.mlx
@pytest.mark.parametrize(
    "act,normalization,p_val,p_arg",
    list(
        itertools.product(
            ["tanh", "square"], ["norm", "component"], [-1, 1], [-1, 1]
        )
    ),
)
def test_upstream_s2_activation_equivariance(
    act: str, normalization: str, p_val: int, p_arg: int
) -> None:
    mx = mlx_backend._require()
    activation = mx.tanh if act == "tanh" else lambda values: values**2
    irreps = _spherical_tensor(3, p_val, p_arg)
    module = e3nn.S2Activation(
        irreps,
        activation,
        120,
        normalization=normalization,
        lmax_out=6,
        random_rot=True,
    )
    assert "S2Activation" in repr(module)
    values = mx.random.normal(shape=(irreps.dim,))

    # Upstream's assert_equivariant performs ten random trials and checks both
    # proper rotations and a rotation composed with inversion.
    for _ in range(10):
        angles = e3nn.rand_angles()
        for inversion in (0, 1):
            d_in = e3nn.irreps_wigner_d(module.irreps_in, *angles, k=inversion)
            d_out = e3nn.irreps_wigner_d(module.irreps_out, *angles, k=inversion)
            actual = module(values @ mx.swapaxes(d_in, -1, -2))
            expected = module(values) @ mx.swapaxes(d_out, -1, -2)
            assert _max_abs(actual - expected) < 0.032
