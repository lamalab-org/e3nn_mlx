"""Correctness and differentiation checks for generated Metal operations."""

from __future__ import annotations

import pytest

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx import FullTensorProduct, IrrepsArray, scatter_sum, spherical_harmonics
from e3nn_mlx.compat import mlx_metal_available


pytestmark = pytest.mark.skipif(
    not mlx_metal_available(), reason="generated kernels require the Metal backend"
)


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


@pytest.mark.mlx
@pytest.mark.parametrize("indices", [[-1, 0], [0, 2]])
def test_scatter_kernel_rejects_indices_outside_output(indices):
    mx = mlx_backend._require()
    source = mx.ones((2, 4), dtype=mx.float32)
    index = mx.array(indices, dtype=mx.int32)
    with pytest.raises(ValueError, match="0 <= index < dim_size"):
        scatter_sum(source, index, 2, use_custom_kernel=True)


@pytest.mark.mlx
def test_empty_inputs_bypass_zero_grid_metal_kernels():
    mx = mlx_backend._require()
    empty_vectors = mx.zeros((0, 3), dtype=mx.float32)
    harmonics = spherical_harmonics(
        [0, 1, 2], empty_vectors, use_custom_kernel=True
    )
    assert harmonics.shape == (0, 9)

    empty_source = mx.zeros((0, 4), dtype=mx.float32)
    empty_index = mx.zeros((0,), dtype=mx.int32)
    scattered = scatter_sum(
        empty_source, empty_index, 3, use_custom_kernel=True
    )
    assert scattered.shape == (3, 4)


@pytest.mark.mlx
def test_unweighted_tensor_product_kernel_second_derivative_matches_general():
    mx = mlx_backend._require()
    kernel = FullTensorProduct("1o", "1o", use_custom_kernel=True)
    general = FullTensorProduct("1o", "1o", use_custom_kernel=False)
    assert kernel._metal_operation is not None
    left = mx.array([[0.2, 0.4, -0.3]], dtype=mx.float32)
    right = mx.array([[-0.5, 0.1, 0.7]], dtype=mx.float32)

    def loss(module, value):
        output = module(IrrepsArray("1o", value), IrrepsArray("1o", right))
        return mx.sum(output.array**2)

    def second_derivative(module):
        first = mx.grad(lambda value: loss(module, value))
        return mx.grad(lambda value: mx.sum(first(value)))(left)

    assert _maximum_error(
        second_derivative(kernel), second_derivative(general)
    ) < 2e-5
