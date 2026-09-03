from __future__ import annotations

import math

import pytest

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray
from e3nn_mlx.nn_gate import Gate
from e3nn_mlx.nn_linear import Linear
from e3nn_mlx.nn_norm import Norm
from e3nn_mlx.ops_reduce import cross, dot, norm


@pytest.mark.mlx
def test_linear_mixes_multiplicities_per_irrep() -> None:
    linear = Linear("2x0e + 1o", "1x0e + 1o", bias=True)
    weight = mlx_backend.asarray(
        [
            2.0 * math.sqrt(2.0),
            -math.sqrt(2.0),
            1.0,
        ]
    )
    bias = mlx_backend.asarray([0.5])
    array = IrrepsArray("2x0e + 1o", mlx_backend.asarray([[1.0, 3.0, 4.0, 5.0, 6.0]]))
    out = linear(array, weight=weight, bias=bias)
    assert out.irreps == linear.irreps_out
    assert out.array.tolist()[0] == pytest.approx([2.0 * 1.0 - 1.0 * 3.0 + 0.5, 4.0, 5.0, 6.0])


@pytest.mark.mlx
def test_linear_gradient_exists() -> None:
    mx = mlx_backend._require()
    linear = Linear("1o", "1o")
    array = IrrepsArray("1o", mlx_backend.asarray([[0.2, 0.3, 0.4]]))

    def loss_fn(weight):
        out = linear(array, weight=weight)
        return mx.sum(out.array * out.array)

    grad = mx.grad(loss_fn)(linear.weight)
    mx.eval(grad)
    assert grad.shape == linear.weight.shape


@pytest.mark.mlx
def test_norm_and_dot_semantics() -> None:
    array = IrrepsArray("0e + 1o", mlx_backend.asarray([[2.0, 3.0, 4.0, 0.0]]))
    norms = norm(array)
    dots = dot(array, array)
    assert str(norms.irreps) == "1x0e+1x0e"
    assert norms.array.tolist()[0][0] == 2.0
    assert abs(norms.array.tolist()[0][1] - 5.0) < 1e-6
    assert dots.array.tolist()[0] == [4.0, 25.0]


@pytest.mark.mlx
def test_cross_returns_axial_vector() -> None:
    left = IrrepsArray("1o", mlx_backend.asarray([[1.0, 0.0, 0.0]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[0.0, 1.0, 0.0]]))
    out = cross(left, right)
    assert str(out.irreps) == "1x1e"
    assert out.array.tolist()[0] == [0.0, 0.0, 1.0]


@pytest.mark.mlx
def test_gate_applies_scalar_and_gate_activations() -> None:
    gate = Gate("0e", "1x0e", "1x1o")
    array = IrrepsArray("0e + 0e + 1o", mlx_backend.asarray([[2.0, 0.0, 1.0, 2.0, 3.0]]))
    out = gate(array)
    gated = out.array.tolist()[0]
    assert len(gated) == 4
    assert abs(gated[0]) < 1.0
    assert gated[1:] == [0.5, 1.0, 1.5]


@pytest.mark.mlx
def test_norm_module_matches_function() -> None:
    module = Norm(per_irrep=False)
    array = IrrepsArray("1o", mlx_backend.asarray([[3.0, 4.0, 0.0]]))
    out = module(array)
    assert out.array.tolist()[0] == [5.0]
