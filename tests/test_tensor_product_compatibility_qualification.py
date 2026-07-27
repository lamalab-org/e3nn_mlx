"""Deterministic compatibility checks for tensor-product edge cases."""

from __future__ import annotations

from math import sqrt

import pytest

import e3nn_mlx as o3
from e3nn_mlx.backend import mlx_backend


def _array(irreps, values):
    return o3.IrrepsArray(irreps, values)


def _maximum_error(first, second) -> float:
    mx = mlx_backend._require()
    if first.size == 0:
        return 0.0
    return float(mx.max(mx.abs(first - second)))


@pytest.mark.mlx
@pytest.mark.parametrize(
    ("irreps_in1", "instruction_index", "expected"),
    [
        ("0x0e + 1x0e + 1x0e", 1, 6.0),
        ("1x0e + 0x1o + 1x0e", 2, 21.0),
        ("0x1o + 0x0e + 1x0e + 0x2e + 1x0e", 2, 6.0),
    ],
)
def test_explicit_instruction_indices_survive_zero_multiplicity_removal(
    irreps_in1, instruction_index, expected
) -> None:
    mx = mlx_backend._require()
    product = o3.TensorProduct(
        irreps_in1,
        "1x0e",
        "1x0e",
        [(instruction_index, 0, 0, "uuu", False)],
        irrep_normalization="none",
        path_normalization="none",
        internal_weights=False,
        use_custom_kernel=False,
    )
    output = product(
        _array(product.irreps_in1, mx.array([[2.0, 7.0]])),
        _array(product.irreps_in2, mx.array([[3.0]])),
    )
    assert output.array.tolist() == [[expected]]


@pytest.mark.mlx
def test_zero_multiplicity_edge_cases_are_explicit() -> None:
    mx = mlx_backend._require()
    empty = o3.TensorProduct(
        "0x0e",
        "0x0e + 0x1o",
        "0x0e",
        [],
        internal_weights=False,
    )
    result = empty(
        _array("", mx.zeros((2, 0))),
        _array("", mx.zeros((2, 0))),
    )
    assert result.shape == (2, 0)

    vector = o3.TensorProduct(
        "0x0e + 1x1o",
        "1x0e",
        "1x1o",
        [(1, 0, 0, "uuu", False)],
        irrep_normalization="none",
        path_normalization="none",
        internal_weights=False,
    )
    values = mx.array([[0.17, -0.31, 0.73]])
    clean_vector = o3.TensorProduct(
        "1x1o",
        "1x0e",
        "1x1o",
        [(0, 0, 0, "uuu", False)],
        irrep_normalization="none",
        path_normalization="none",
        internal_weights=False,
    )
    assert _maximum_error(
        vector(_array("1o", values), _array("0e", mx.array([[2.0]]))).array,
        clean_vector(
            _array("1o", values),
            _array("0e", mx.array([[2.0]])),
        ).array,
    ) == 0.0

    repeated = o3.TensorProduct(
        "1x1o + 0x0e + 1x1o",
        "1x0e",
        "1x1o",
        [(2, 0, 0, "uuu", False)],
        irrep_normalization="none",
        path_normalization="none",
        internal_weights=False,
    )
    both_vectors = mx.array([[0.17, -0.31, 0.73, 1.11, -1.37, 0.29]])
    clean_repeated = o3.TensorProduct(
        "1x1o",
        "1x0e",
        "1x1o",
        [(0, 0, 0, "uuu", False)],
        irrep_normalization="none",
        path_normalization="none",
        internal_weights=False,
    )
    expected = clean_repeated(
        _array("1o", both_vectors[:, 3:]),
        _array("0e", mx.array([[3.0]])),
    ).array
    actual = repeated(
        _array(repeated.irreps_in1, both_vectors),
        _array("0e", mx.array([[3.0]])),
    ).array
    assert _maximum_error(actual, expected) == 0.0


@pytest.mark.mlx
@pytest.mark.parametrize(
    ("irreps_in1", "irreps_in2", "irreps_out", "instruction"),
    [
        ("0x0e + 1x0e", "1x0e", "1x0e", (0, 0, 0, "uuu", False)),
        ("1x0e", "0x0e + 1x0e", "1x0e", (0, 0, 0, "uuu", False)),
        ("1x0e", "1x0e", "0x0e + 1x0e", (0, 0, 0, "uuu", False)),
    ],
)
def test_instruction_referencing_zero_multiplicity_is_rejected(
    irreps_in1, irreps_in2, irreps_out, instruction
) -> None:
    with pytest.raises(ValueError, match="zero-multiplicity"):
        o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            [instruction],
            internal_weights=False,
        )


@pytest.mark.mlx
@pytest.mark.parametrize("compile_left_right", [False, True])
@pytest.mark.parametrize("use_custom_kernel", [False, True])
def test_multiple_unweighted_instructions_sum_into_one_output(
    compile_left_right, use_custom_kernel
) -> None:
    mx = mlx_backend._require()
    product = o3.TensorProduct(
        "1x0e + 1x0e",
        "1x0e",
        "1x0e",
        [
            (0, 0, 0, "uuu", False),
            (1, 0, 0, "uuu", False),
        ],
        irrep_normalization="none",
        path_normalization="none",
        internal_weights=False,
        compile_left_right=compile_left_right,
        use_custom_kernel=use_custom_kernel,
    )
    output = product(
        _array(product.irreps_in1, mx.array([[2.0, 7.0]])),
        _array("0e", mx.array([[3.0]])),
    )
    assert output.shape == (1, 1)
    assert output.array.tolist() == [[27.0]]


@pytest.mark.mlx
def test_repeated_output_accumulation_covers_disabled_mixed_and_vector_paths() -> None:
    mx = mlx_backend._require()
    disabled = o3.TensorProduct(
        "1x0e + 1x0e",
        "1x0e",
        "1x0e",
        [
            (0, 0, 0, "uuu", False, 1.0),
            (1, 0, 0, "uuu", False, 0.0),
        ],
        irrep_normalization="none",
        path_normalization="none",
        internal_weights=False,
    )
    left = _array(disabled.irreps_in1, mx.array([[2.0, 7.0]]))
    right = _array("0e", mx.array([[3.0]]))
    assert disabled(left, right).array.tolist() == [[6.0]]

    mixed = o3.TensorProduct(
        "1x0e + 1x0e",
        "1x0e",
        "1x0e",
        [
            (0, 0, 0, "uuu", False),
            (1, 0, 0, "uvw", True),
        ],
        irrep_normalization="none",
        path_normalization="none",
        internal_weights=False,
    )
    assert mixed(left, right, mx.array([13.0])).array.tolist() == [[279.0]]

    vector = o3.TensorProduct(
        "1x1o + 1x1o",
        "1x1o",
        "1x0e",
        [
            (0, 0, 0, "uuu", False),
            (1, 0, 0, "uuu", False),
        ],
        irrep_normalization="none",
        path_normalization="none",
        internal_weights=False,
        use_custom_kernel=False,
    )
    vector_left = _array(
        vector.irreps_in1,
        mx.array([[0.17, -0.31, 0.73, 1.11, -1.37, 0.29]]),
    )
    vector_right = _array("1o", mx.array([[-0.41, 0.59, 0.97]]))
    expected = (
        mx.sum(vector_left.array[:, :3] * vector_right.array, axis=-1)
        + mx.sum(vector_left.array[:, 3:] * vector_right.array, axis=-1)
    ) / sqrt(3.0)
    assert _maximum_error(vector(vector_left, vector_right).array[:, 0], expected) < 2e-6


@pytest.mark.mlx
def test_unshared_weights_require_a_leading_dimension() -> None:
    mx = mlx_backend._require()
    product = o3.TensorProduct(
        "1x0e",
        "1x0e",
        "1x0e",
        [(0, 0, 0, "uvw", True)],
        internal_weights=False,
        shared_weights=False,
        irrep_normalization="none",
        path_normalization="none",
    )
    left = _array("0e", mx.array([[2.0], [7.0], [11.0]]))
    right = _array("0e", mx.array([[3.0], [5.0], [13.0]]))
    with pytest.raises(ValueError, match="batch dimension"):
        product(left, right, mx.array([1.0]))

    singleton = product(left, right, mx.array([[2.0]])).array
    per_example = product(
        left,
        right,
        mx.array([[2.0], [3.0], [5.0]]),
    ).array
    assert singleton.shape == (3, 1)
    assert singleton.tolist() == [[12.0], [70.0], [286.0]]
    assert per_example.tolist() == [[12.0], [105.0], [715.0]]

    with pytest.raises(ValueError, match="Expected unshared weight shape"):
        product(left, right, mx.ones((3, 1, 2)))
    with pytest.raises(ValueError, match="broadcastable"):
        product(left, right, mx.ones((4, 1)))

    empty = product(
        _array("0e", mx.zeros((0, 1))),
        _array("0e", mx.zeros((0, 1))),
        mx.zeros((0, 1)),
    )
    assert empty.shape == (0, 1)


@pytest.mark.mlx
def test_shared_tensor_product_weights_are_not_materialized_per_item(
    monkeypatch,
) -> None:
    mx = mlx_backend._require()
    product = o3.FullyConnectedTensorProduct(
        "4x0e + 4x1o",
        "4x0e + 4x1o",
        "4x0e + 4x1o",
        use_custom_kernel=False,
    )
    left = _array(product.irreps_in1, mx.ones((8, product.irreps_in1.dim)))
    right = _array(product.irreps_in2, mx.ones((8, product.irreps_in2.dim)))
    shared_weight = product.weight
    materialized_shapes = []
    original_broadcast_to = mx.broadcast_to

    def record_broadcast(array, shape, *args, **kwargs):
        if array is shared_weight:
            materialized_shapes.append(tuple(shape))
        return original_broadcast_to(array, shape, *args, **kwargs)

    monkeypatch.setattr(mx, "broadcast_to", record_broadcast)
    output = product(left, right)
    mx.eval(output.array)

    assert output.shape == (8, product.irreps_out.dim)
    assert materialized_shapes == []


@pytest.mark.mlx
def test_compact_shared_weight_gradient_matches_summed_per_item_gradient() -> None:
    mx = mlx_backend._require()
    product = o3.FullyConnectedTensorProduct(
        "2x0e + 2x1o",
        "2x0e + 2x1o",
        "2x0e + 2x1o",
        use_custom_kernel=False,
    )
    left = mx.arange(24, dtype=mx.float32).reshape(3, 8) / 17.0 - 0.4
    right = mx.arange(24, dtype=mx.float32).reshape(3, 8) / 19.0 - 0.6
    shared_weight = product.weight
    per_item_weight = mx.broadcast_to(
        shared_weight,
        (left.shape[0], product.weight_numel),
    )

    def loss(weight):
        return mx.sum(product._general_call_arrays(left, right, weight) ** 2)

    shared_output = product._general_call_arrays(left, right, shared_weight)
    per_item_output = product._general_call_arrays(left, right, per_item_weight)
    shared_gradient = mx.grad(loss)(shared_weight)
    per_item_gradient = mx.grad(loss)(per_item_weight)

    assert _maximum_error(shared_output, per_item_output) < 2e-6
    assert _maximum_error(
        shared_gradient,
        mx.sum(per_item_gradient, axis=0),
    ) < 2e-5


@pytest.mark.mlx
def test_mixed_weighted_instruction_views_match_forward_and_weight_lists() -> None:
    mx = mlx_backend._require()
    product = o3.TensorProduct(
        "2x0e + 2x0e + 2x0e",
        "2x0e",
        "2x0e + 3x0e + 2x0e",
        [
            (0, 0, 0, "uuu", False),
            (1, 0, 1, "uvw", True),
            (2, 0, 2, "uuu", False),
            (0, 0, 0, "uvu", True),
        ],
        internal_weights=False,
        irrep_normalization="none",
        path_normalization="none",
        use_custom_kernel=False,
    )
    weight = mx.arange(1, product.weight_numel + 1, dtype=mx.float32)
    assert product.weight_numel == 16
    first = product.weight_view_for_instruction(1, weight)
    second = product.weight_view_for_instruction(3, weight)
    assert first.shape == (2, 2, 3)
    assert second.shape == (2, 2)
    assert first.reshape(-1).tolist() == list(range(1, 13))
    assert second.reshape(-1).tolist() == list(range(13, 17))
    views = list(product.weight_views(weight))
    assert len(views) == 2
    assert _maximum_error(views[0], first) == 0.0
    assert _maximum_error(views[1], second) == 0.0

    left = _array(
        product.irreps_in1,
        mx.array([[0.17, -0.31, 0.73, 1.11, -1.37, 0.29]]),
    )
    right = _array(product.irreps_in2, mx.array([[0.41, -0.59]]))
    flattened = product(left, right, weight).array
    per_instruction = product(left, right, [first, second]).array
    assert _maximum_error(flattened, per_instruction) == 0.0


@pytest.mark.mlx
def test_unshared_weight_broadcast_matrix_matches_flattened_execution() -> None:
    mx = mlx_backend._require()
    product = o3.TensorProduct(
        "2x0e",
        "3x0e",
        "2x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
        use_custom_kernel=False,
    )
    left = mx.arange(12, dtype=mx.float32).reshape(2, 1, 3, 2) / 7.0
    right = mx.arange(12, dtype=mx.float32).reshape(1, 4, 1, 3) / 11.0
    weights = mx.arange(
        2 * 4 * 3 * product.weight_numel,
        dtype=mx.float32,
    ).reshape(2, 4, 3, product.weight_numel) / 13.0
    output = product(
        _array(product.irreps_in1, left),
        _array(product.irreps_in2, right),
        weights,
    ).array
    assert output.shape == (2, 4, 3, product.irreps_out.dim)

    flat_left = mx.broadcast_to(left, (2, 4, 3, 2)).reshape(-1, 2)
    flat_right = mx.broadcast_to(right, (2, 4, 3, 3)).reshape(-1, 3)
    expected = product(
        _array(product.irreps_in1, flat_left),
        _array(product.irreps_in2, flat_right),
        weights.reshape(-1, product.weight_numel),
    ).array.reshape(output.shape)
    assert _maximum_error(output, expected) < 2e-6


@pytest.mark.mlx
@pytest.mark.parametrize("use_custom_kernel", [False, True])
def test_weight_dtype_policy_is_consistent_across_execution_paths(
    use_custom_kernel,
) -> None:
    mx = mlx_backend._require()
    product = o3.TensorProduct(
        "1x0e",
        "1x0e",
        "1x0e",
        [(0, 0, 0, "uvw", True)],
        internal_weights=False,
        use_custom_kernel=use_custom_kernel,
    )
    left = _array("0e", mx.ones((2, 1), dtype=mx.float32))
    right = _array("0e", mx.ones((2, 1), dtype=mx.float32))
    with pytest.raises(TypeError, match="weights must match input dtype"):
        product(left, right, mx.ones((1,), dtype=mx.float16))

    half_product = o3.TensorProduct(
        "1x0e",
        "1x0e",
        "1x0e",
        [(0, 0, 0, "uvw", True)],
        internal_weights=False,
        use_custom_kernel=use_custom_kernel,
    )
    half_output = half_product(
        _array("0e", mx.ones((2, 1), dtype=mx.float16)),
        _array("0e", mx.ones((2, 1), dtype=mx.float16)),
        mx.ones((1,), dtype=mx.float16),
    )
    assert half_output.dtype == mx.float16


@pytest.mark.mlx
def test_constructor_rejects_upstream_invalid_uvw_and_static_configurations() -> None:
    with pytest.raises(ValueError, match="requires weights"):
        o3.TensorProduct(
            "1x0e",
            "1x0e",
            "1x0e",
            [(0, 0, 0, "uvw", False)],
            internal_weights=False,
        )
    with pytest.raises(IndexError, match="out of range"):
        o3.TensorProduct(
            "1x0e",
            "1x0e",
            "1x0e",
            [(1, 0, 0, "uuu", False)],
            internal_weights=False,
        )
    with pytest.raises(ValueError, match="parity mismatch"):
        o3.TensorProduct(
            "1x0e",
            "1x0e",
            "1x0o",
            [(0, 0, 0, "uuu", False)],
            internal_weights=False,
        )
    with pytest.raises(ValueError, match="selection rule"):
        o3.TensorProduct(
            "1x0e",
            "1x0e",
            "1x1e",
            [(0, 0, 0, "uuu", False)],
            internal_weights=False,
        )
    with pytest.raises(ValueError, match="multiplicit"):
        o3.TensorProduct(
            "2x0e",
            "2x0e",
            "1x0e",
            [(0, 0, 0, "uuu", False)],
            internal_weights=False,
        )
