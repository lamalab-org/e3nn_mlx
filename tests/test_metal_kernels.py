"""Correctness and differentiation checks for generated Metal operations."""

from __future__ import annotations

import pytest

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx import scatter_sum, spherical_harmonics


def _maximum_error(first, second) -> float:
    return float(abs(first - second).max())


@pytest.mark.mlx
def test_spherical_harmonics_kernel_matches_general_forward_gradient_and_hessian():
    mx = mlx_backend._require()
    vectors = mx.array([[0.2, -0.3, 0.7], [-0.4, 0.1, 0.6]], dtype=mx.float32)
    coefficients = mx.arange(32, dtype=mx.float32).reshape(2, 16) / 17.0

    def output(value, enabled):
        return spherical_harmonics(
            [0, 1, 2, 3],
            value,
            normalize=True,
            use_custom_kernel=enabled,
        )

    assert _maximum_error(output(vectors, True), output(vectors, False)) < 2e-5

    # MLX preserves the positional-argument tree for custom VJPs. For a
    # one-argument custom function the callback receives the array directly,
    # not a one-tuple. Exercise VJP explicitly so a tuple-unpacking regression
    # cannot hide behind mx.grad.
    _, (kernel_vjp,) = mx.vjp(
        lambda value: output(value, True),
        (vectors,),
        (coefficients,),
    )
    _, (general_vjp,) = mx.vjp(
        lambda value: output(value, False),
        (vectors,),
        (coefficients,),
    )
    assert _maximum_error(kernel_vjp, general_vjp) < 2e-4

    def loss(value, enabled):
        residual = output(value, enabled) - coefficients
        return mx.sum(residual * residual)

    kernel_gradient = mx.grad(lambda value: loss(value, True))(vectors)
    general_gradient = mx.grad(lambda value: loss(value, False))(vectors)
    assert _maximum_error(kernel_gradient, general_gradient) < 2e-4

    direction = mx.array([[0.3, 0.1, -0.2], [-0.1, 0.4, 0.2]])
    kernel_hessian_vector = mx.grad(
        lambda value: mx.sum(mx.grad(lambda x: loss(x, True))(value) * direction)
    )(vectors)
    general_hessian_vector = mx.grad(
        lambda value: mx.sum(mx.grad(lambda x: loss(x, False))(value) * direction)
    )(vectors)
    assert _maximum_error(kernel_hessian_vector, general_hessian_vector) < 2e-3


@pytest.mark.mlx
def test_scatter_kernel_matches_general_forward_gradient_and_hessian():
    mx = mlx_backend._require()
    source = mx.arange(24, dtype=mx.float32).reshape(6, 4) / 11.0
    index = mx.array([0, 2, 1, 2, 0, 1], dtype=mx.int32)

    def output(value, enabled):
        return scatter_sum(value, index, 3, use_custom_kernel=enabled)

    assert _maximum_error(output(source, True), output(source, False)) < 1e-6

    def loss(value, enabled):
        result = output(value, enabled)
        return mx.sum(result * result)

    kernel_gradient = mx.grad(lambda value: loss(value, True))(source)
    general_gradient = mx.grad(lambda value: loss(value, False))(source)
    assert _maximum_error(kernel_gradient, general_gradient) < 1e-6
    kernel_second = mx.grad(
        lambda value: mx.sum(mx.grad(lambda x: loss(x, True))(value))
    )(source)
    general_second = mx.grad(
        lambda value: mx.sum(mx.grad(lambda x: loss(x, False))(value))
    )(source)
    assert _maximum_error(kernel_second, general_second) < 1e-6
