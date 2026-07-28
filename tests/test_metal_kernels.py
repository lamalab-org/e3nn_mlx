"""Correctness and differentiation checks for generated Metal operations."""

from __future__ import annotations

import pytest

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx import (
    FullyConnectedTensorProduct,
    FullTensorProduct,
    IrrepsArray,
    TensorProduct,
    scatter_sum,
    spherical_harmonics,
)
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


def _channel_tensor_product(*, use_custom_kernel):
    return TensorProduct(
        "2x1o",
        "3x1o",
        "2x0e + 2x1e + 2x2e",
        [
            (0, 0, 0, "uvu", True),
            (0, 0, 1, "uvu", True),
            (0, 0, 2, "uvu", True),
        ],
        internal_weights=False,
        shared_weights=False,
        use_custom_kernel=use_custom_kernel,
    )


@pytest.mark.mlx
def test_weighted_channel_kernel_matches_general_through_nested_derivatives():
    mx = mlx_backend._require()
    kernel = _channel_tensor_product(use_custom_kernel=True)
    general = _channel_tensor_product(use_custom_kernel=False)
    assert kernel._metal_kernel_kind == "channel_uvu"
    assert kernel._metal_operation is not None

    left = mx.arange(18, dtype=mx.float32).reshape(3, 6) / 13.0 - 0.4
    right = mx.arange(27, dtype=mx.float32).reshape(3, 9) / 17.0 - 0.6
    weight = mx.arange(54, dtype=mx.float32).reshape(3, 18) / 19.0 - 0.7
    cotangent = (
        mx.arange(54, dtype=mx.float32).reshape(3, 18) / 23.0 - 0.5
    )

    def apply(module, first, second, value):
        return module._call_arrays(first, second, value)

    kernel_output = apply(kernel, left, right, weight)
    general_output = apply(general, left, right, weight)
    assert _maximum_error(kernel_output, general_output) < 3e-5

    _, kernel_vjp = mx.vjp(
        lambda first, second, value: apply(
            kernel, first, second, value
        ),
        (left, right, weight),
        (cotangent,),
    )
    _, general_vjp = mx.vjp(
        lambda first, second, value: apply(
            general, first, second, value
        ),
        (left, right, weight),
        (cotangent,),
    )
    for actual, expected in zip(kernel_vjp, general_vjp, strict=True):
        assert _maximum_error(actual, expected) < 2e-4

    tangents = (
        mx.ones_like(left) * 0.07,
        mx.ones_like(right) * -0.03,
        mx.ones_like(weight) * 0.11,
    )
    with pytest.raises(ValueError, match="CustomKernel"):
        mx.jvp(
            lambda first, second, value: apply(
                kernel, first, second, value
            ),
            (left, right, weight),
            tangents,
        )
    _, (kernel_jvp,) = mx.jvp(
        kernel.differentiable_arrays,
        (left, right, weight),
        tangents,
    )
    _, (general_jvp,) = mx.jvp(
        lambda first, second, value: apply(
            general, first, second, value
        ),
        (left, right, weight),
        tangents,
    )
    assert _maximum_error(kernel_jvp, general_jvp) < 2e-4

    def loss(module, first, value):
        output = apply(module, first, right, value)
        return mx.sum(output * output)

    direction = mx.arange(left.size, dtype=mx.float32).reshape(left.shape) / 29.0

    def hessian_vector(module):
        first = mx.grad(lambda x: loss(module, x, weight))
        return mx.grad(lambda x: mx.sum(first(x) * direction))(left)

    assert _maximum_error(
        hessian_vector(kernel), hessian_vector(general)
    ) < 2e-3

    def mixed_position_weight(module):
        position_gradient = mx.grad(lambda x, w: loss(module, x, w), argnums=0)
        return mx.grad(
            lambda w: mx.sum(position_gradient(left, w) * direction)
        )(weight)

    assert _maximum_error(
        mixed_position_weight(kernel),
        mixed_position_weight(general),
    ) < 2e-3


@pytest.mark.mlx
def test_weighted_channel_kernel_backward_is_numerically_stable():
    mx = mlx_backend._require()
    product = _channel_tensor_product(use_custom_kernel=True)
    left = mx.arange(24, dtype=mx.float32).reshape(4, 6) / 31.0
    right = mx.arange(36, dtype=mx.float32).reshape(4, 9) / 37.0
    weight = mx.arange(72, dtype=mx.float32).reshape(4, 18) / 41.0

    def gradients():
        _, values = mx.vjp(
            product._call_arrays,
            (left, right, weight),
            (mx.ones((4, 18), dtype=mx.float32),),
        )
        mx.eval(*values)
        return values

    reference = gradients()
    for _ in range(8):
        for actual, expected in zip(gradients(), reference, strict=True):
            assert _maximum_error(actual, expected) < 2e-5


@pytest.mark.mlx
def test_scalar_tensor_product_dispatch_boundary_preserves_results():
    mx = mlx_backend._require()
    product = FullTensorProduct("1o", "1o", use_custom_kernel=True)
    assert product._metal_kernel_kind == "scalar_paths"

    left = mx.arange(513 * 3, dtype=mx.float32).reshape(513, 3) / 997.0
    right = mx.arange(513 * 3, dtype=mx.float32).reshape(513, 3) / 991.0
    custom = product._try_metal(left[:512], right[:512], None)
    fallback = product._try_metal(left, right, None)
    assert custom is not None
    assert fallback is None
    expected = product._general_call_arrays(left[:512], right[:512])
    assert _maximum_error(custom, expected) < 3e-5


@pytest.mark.mlx
def test_dense_scalar_tensor_product_dispatch_uses_work_guard():
    mx = mlx_backend._require()
    irreps = "8x0e + 8x1o + 8x2e"
    product = FullyConnectedTensorProduct(
        irreps,
        irreps,
        irreps,
        use_custom_kernel=True,
    )
    assert product._metal_kernel_kind == "scalar_paths"
    assert product._metal_operation is not None

    small_left = mx.ones((16, product.irreps_in1.dim), dtype=mx.float32)
    small_right = mx.ones((16, product.irreps_in2.dim), dtype=mx.float32)
    large_left = mx.ones((64, product.irreps_in1.dim), dtype=mx.float32)
    large_right = mx.ones((64, product.irreps_in2.dim), dtype=mx.float32)

    assert (
        product._metal_dispatch_kind(small_left, small_right, product.weight)
        == "scalar_paths"
    )
    assert product._try_metal(small_left, small_right, product.weight) is not None
    assert product._metal_dispatch_kind(
        large_left, large_right, product.weight
    ) is None
    assert product._try_metal(large_left, large_right, product.weight) is None


@pytest.mark.mlx
def test_tensor_product_dispatch_falls_back_for_rank_and_dtype():
    mx = mlx_backend._require()
    product = FullTensorProduct("1o", "1o", use_custom_kernel=True)
    left = mx.arange(6, dtype=mx.float32).reshape(2, 3) / 7.0
    right = mx.arange(6, dtype=mx.float32).reshape(2, 3) / 11.0

    assert product._try_metal(left, right, None) is not None
    assert product._try_metal(left[None], right[None], None) is None
    rank_three = product(
        IrrepsArray("1o", left[None]),
        IrrepsArray("1o", right[None]),
    ).array
    rank_two = product(
        IrrepsArray("1o", left),
        IrrepsArray("1o", right),
    ).array
    assert _maximum_error(rank_three[0], rank_two) < 2e-6

    left16 = left.astype(mx.float16)
    right16 = right.astype(mx.float16)
    assert product._try_metal(left16, right16, None) is None
    output16 = product(
        IrrepsArray("1o", left16),
        IrrepsArray("1o", right16),
    ).array
    assert output16.dtype == mx.float16
    assert _maximum_error(output16.astype(mx.float32), rank_two) < 2e-3


@pytest.mark.mlx
def test_mixed_weight_ownership_uses_general_tensor_product_path():
    mx = mlx_backend._require()
    requested = TensorProduct(
        "1x0e + 1x0e",
        "1x0e",
        "1x0e",
        [
            (0, 0, 0, "uuu", False),
            (1, 0, 0, "uvw", True),
        ],
        internal_weights=False,
        use_custom_kernel=True,
    )
    general = TensorProduct(
        requested.irreps_in1,
        requested.irreps_in2,
        requested.irreps_out,
        [
            (0, 0, 0, "uuu", False),
            (1, 0, 0, "uvw", True),
        ],
        internal_weights=False,
        use_custom_kernel=False,
    )
    assert requested._metal_operation is None
    left = mx.array([[2.0, 7.0]], dtype=mx.float32)
    right = mx.array([[3.0]], dtype=mx.float32)
    weight = mx.array([5.0], dtype=mx.float32)
    assert _maximum_error(
        requested._call_arrays(left, right, weight),
        general._call_arrays(left, right, weight),
    ) < 1e-6
