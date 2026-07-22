"""Extensive tests for model graph and radial utilities."""

from __future__ import annotations

from math import exp

import pytest

import e3nn_mlx as e3nn
from e3nn_mlx.backend import mlx_backend


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value))) if value.size else 0.0


@pytest.mark.mlx
def test_soft_unit_step_values_compile_and_higher_gradients() -> None:
    mx = mlx_backend._require()
    values = mx.array([-2.0, -0.1, 0.0, 0.25, 1.0, 2.0])
    output = e3nn.soft_unit_step(values)
    assert output[:3].tolist() == [0.0, 0.0, 0.0]
    assert abs(float(output[4]) - exp(-1.0)) < 2e-7
    compiled = mx.compile(e3nn.soft_unit_step)
    assert _max_abs(compiled(values) - output) < 2e-7

    first = mx.grad(lambda x: mx.sum(e3nn.soft_unit_step(x)))
    second = mx.grad(lambda x: mx.sum(first(x)))
    third = mx.grad(lambda x: mx.sum(second(x)))
    for derivative in (first(values), second(values), third(values)):
        mx.eval(derivative)
        assert bool(mx.all(mx.isfinite(derivative)))
        assert derivative[:3].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.mlx
@pytest.mark.parametrize("basis", ["gaussian", "cosine", "fourier", "bessel", "smooth_finite"])
@pytest.mark.parametrize("cutoff", [True, False])
def test_soft_one_hot_all_bases_compile_shape_dtype_and_gradients(basis: str, cutoff: bool) -> None:
    mx = mlx_backend._require()
    values = mx.linspace(-2.0, 3.0, 20)

    def function(x):
        return e3nn.soft_one_hot_linspace(x, -1.0, 2.0, 5, basis, cutoff)

    output = function(values)
    assert output.shape == (20, 5)
    assert output.dtype == values.dtype
    compiled = mx.compile(function)
    assert _max_abs(compiled(values) - output) < 3e-6
    gradient = mx.grad(lambda x: mx.sum(function(x) ** 2))(values)
    mx.eval(gradient)
    assert bool(mx.all(mx.isfinite(gradient)))


@pytest.mark.mlx
@pytest.mark.parametrize("basis", ["gaussian", "cosine", "fourier", "bessel", "smooth_finite"])
def test_soft_one_hot_cutoff_outside_interval(basis: str) -> None:
    mx = mlx_backend._require()
    values = mx.concatenate([mx.linspace(-2.0, -1.1, 20), mx.linspace(2.1, 3.0, 20)])
    output = e3nn.soft_one_hot_linspace(values, -1.0, 2.0, 5, basis, True)
    if basis == "gaussian":
        assert _max_abs(output) < 0.22
    else:
        assert _max_abs(output) == 0.0


@pytest.mark.mlx
@pytest.mark.parametrize("basis", ["gaussian", "cosine", "fourier", "smooth_finite"])
@pytest.mark.parametrize("cutoff", [True, False])
def test_soft_one_hot_second_moment_normalization(basis: str, cutoff: bool) -> None:
    mx = mlx_backend._require()
    values = mx.linspace(-14.0, 105.0, 50)
    output = e3nn.soft_one_hot_linspace(values, -20.0, 120.0, 12, basis, cutoff)
    norm = mx.sum(output**2, axis=-1)
    assert float(mx.min(norm)) > 0.4
    assert float(mx.max(norm)) < 2.0


@pytest.mark.mlx
def test_soft_one_hot_validation_and_bessel_origin_limit() -> None:
    mx = mlx_backend._require()
    with pytest.raises(ValueError, match="boolean"):
        e3nn.soft_one_hot_linspace(mx.ones((2,)), 0, 1, 2, "gaussian", None)
    with pytest.raises(ValueError, match="positive"):
        e3nn.soft_one_hot_linspace(mx.ones((2,)), 0, 1, 0, "gaussian", True)
    with pytest.raises(ValueError, match="greater"):
        e3nn.soft_one_hot_linspace(mx.ones((2,)), 1, 0, 2, "gaussian", True)
    with pytest.raises(ValueError, match="valid entry"):
        e3nn.soft_one_hot_linspace(mx.ones((2,)), 0, 1, 2, "invalid", True)
    at_origin = e3nn.soft_one_hot_linspace(mx.array([0.0]), 0, 2, 3, "bessel", False)
    assert bool(mx.all(mx.isfinite(at_origin)))


@pytest.mark.mlx
def test_smooth_cutoff_values_compile_and_gradient() -> None:
    mx = mlx_backend._require()
    values = mx.array([-1.0, 0.0, 0.49, 0.5, 0.75, 1.0, 1.5])
    output = e3nn.smooth_cutoff(values)
    assert output[:4].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert abs(float(output[4]) - 0.5) < 2e-7
    assert output[5:].tolist() == [0.0, 0.0]
    assert _max_abs(mx.compile(e3nn.smooth_cutoff)(values) - output) < 2e-7
    gradient = mx.grad(lambda x: mx.sum(e3nn.smooth_cutoff(x)))(values)
    assert bool(mx.all(mx.isfinite(gradient)))


@pytest.mark.mlx
def test_scatter_sum_values_multidimensional_empty_compile_and_gradient() -> None:
    mx = mlx_backend._require()
    source = mx.arange(24, dtype=mx.float32).reshape(4, 2, 3)
    index = mx.array([2, 0, 2, 0], dtype=mx.int32)
    expected = mx.stack([source[1] + source[3], mx.zeros((2, 3)), source[0] + source[2]])
    output = e3nn.scatter_sum(source, index, dim_size=3)
    assert _max_abs(output - expected) == 0.0
    compiled = mx.compile(lambda x, i: e3nn.scatter_sum(x, i, 3))
    assert _max_abs(compiled(source, index) - expected) == 0.0
    gradient = mx.grad(lambda x: mx.sum(e3nn.scatter_sum(x, index, 3) ** 2))(source)
    mx.eval(gradient)
    assert bool(mx.all(mx.isfinite(gradient)))
    safe = e3nn.scatter_sum(source, index, 3, jvp_safe=True)
    assert _max_abs(safe - expected) == 0.0
    safe_gradient = mx.grad(
        lambda x: mx.sum(e3nn.scatter_sum(x, index, 3, jvp_safe=True) ** 2)
    )(source)
    assert _max_abs(safe_gradient - gradient) == 0.0
    (jvp_output,), (jvp_tangent,) = mx.jvp(
        lambda x: e3nn.scatter_sum(x, index, 3, jvp_safe=True),
        (source,),
        (mx.ones_like(source),),
    )
    assert _max_abs(jvp_output - expected) == 0.0
    assert _max_abs(
        jvp_tangent
        - e3nn.scatter_sum(mx.ones_like(source), index, 3, jvp_safe=True)
    ) == 0.0
    compiled_safe = mx.compile(
        lambda x: e3nn.scatter_sum(x, index, 3, jvp_safe=True)
    )
    assert _max_abs(compiled_safe(source) - expected) == 0.0
    empty = e3nn.scatter_sum(mx.zeros((0, 4)), mx.zeros((0,), dtype=mx.int32), 2)
    assert empty.shape == (2, 4)


@pytest.mark.mlx
def test_scatter_sum_validation() -> None:
    mx = mlx_backend._require()
    with pytest.raises(ValueError, match="one entry"):
        e3nn.scatter_sum(mx.ones((2, 3)), mx.array([0], dtype=mx.int32), 1)
    with pytest.raises(TypeError, match="integer"):
        e3nn.scatter_sum(mx.ones((2, 3)), mx.array([0.0, 0.0]), 1)
    with pytest.raises(ValueError, match="requires use_custom_kernel=False"):
        e3nn.scatter_sum(
            mx.ones((2, 3)),
            mx.array([0, 0], dtype=mx.int32),
            1,
            use_custom_kernel=True,
            jvp_safe=True,
        )
    with pytest.raises(ValueError, match="0 <= index < dim_size"):
        e3nn.scatter_sum(
            mx.ones((2, 3)),
            mx.array([0, 2], dtype=mx.int32),
            2,
            jvp_safe=True,
        )


@pytest.mark.mlx
def test_radius_graph_exact_batching_symmetry_and_rigid_motion_invariance() -> None:
    mx = mlx_backend._require()
    positions = mx.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [2.4, 0.0, 0.0]]
    )
    batch = mx.array([0, 0, 1, 1], dtype=mx.int32)
    expected = [[0, 1, 2, 3], [1, 0, 3, 2]]
    assert e3nn.radius_graph(positions, 1.0, batch).tolist() == expected
    assert e3nn.radius_graph(positions, 1.0).tolist() == expected

    rotation = e3nn.rand_matrix()
    translation = mx.array([3.0, -2.0, 7.0])
    moved = positions @ mx.swapaxes(rotation, -1, -2) + translation
    assert e3nn.radius_graph(moved, 1.0, batch).tolist() == expected


@pytest.mark.mlx
def test_radius_graph_strict_boundary_coincident_empty_and_validation() -> None:
    mx = mlx_backend._require()
    positions = mx.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert e3nn.radius_graph(positions, 1.0).shape == (2, 0)
    assert e3nn.radius_graph(mx.zeros((0, 3)), 1.0).shape == (2, 0)
    with pytest.raises(ValueError, match="shape"):
        e3nn.radius_graph(mx.zeros((2, 2)), 1.0)
    with pytest.raises(ValueError, match="positive"):
        e3nn.radius_graph(mx.zeros((2, 3)), 0.0)
    with pytest.raises(TypeError, match="integer"):
        e3nn.radius_graph(mx.zeros((2, 3)), 1.0, mx.zeros((2,)))
