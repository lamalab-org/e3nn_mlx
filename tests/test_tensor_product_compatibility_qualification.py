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
