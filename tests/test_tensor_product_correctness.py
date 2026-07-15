from __future__ import annotations

import math
from typing import Any

import pytest

from e3nn_core.cg import clebsch_gordan
from e3nn_core.irreps import Irreps
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray
from e3nn_mlx.ops_tp import ElementwiseTensorProduct, FullTensorProduct, FullyConnectedTensorProduct, TensorProduct, TensorSquare

pytestmark = pytest.mark.mlx


def _flatten_rows(value: Any) -> list[list[float]]:
    rows = value.tolist()
    if rows and not isinstance(rows[0], list):
        return [rows]
    return rows


def _max_diff(actual: Any, expected: list[list[float]]) -> float:
    actual_rows = _flatten_rows(actual)
    return max(
        abs(a - b)
        for row_a, row_b in zip(actual_rows, expected, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
    )


def _regroup_flat_rows(rows: list[list[float]], source: str, target: str) -> list[list[float]]:
    source_irreps = Irreps(source)
    target_irreps = Irreps(target)
    source_slices = source_irreps.slices()
    grouped_by_irrep: dict[Any, list[tuple[int, slice]]] = {}
    for part, chunk_slice in zip(source_irreps, source_slices, strict=True):
        grouped_by_irrep.setdefault(part.ir, []).append((part.mul, chunk_slice))

    regrouped = []
    for row in rows:
        out_row = []
        for part in target_irreps:
            pieces = []
            for mul, chunk_slice in grouped_by_irrep[part.ir]:
                width = chunk_slice.stop - chunk_slice.start
                dim = width // mul
                block = row[chunk_slice]
                for i in range(mul):
                    pieces.extend(block[i * dim : (i + 1) * dim])
            assert len(pieces) == part.mul * part.ir.dim
            out_row.extend(pieces)
        regrouped.append(out_row)
    return regrouped


def _chunk_rows(array: IrrepsArray) -> list[list[list[float]]]:
    return [_flatten_rows(chunk) for chunk in array.chunk_arrays()]


def _reshape_weight_rows(weight: Any, shape: tuple[int, ...], batch_size: int, shared_weights: bool) -> list[Any]:
    rows = _flatten_rows(weight)
    if shared_weights:
        data = rows[0]
        return [_reshape_flat(data, shape) for _ in range(batch_size)]
    return [_reshape_flat(row, shape) for row in rows]


def _reshape_flat(values: list[float], shape: tuple[int, ...]) -> Any:
    iterator = iter(values)

    def build(axis: int) -> Any:
        if axis == len(shape):
            return float(next(iterator))
        return [build(axis + 1) for _ in range(shape[axis])]

    out = build(0)
    try:
        next(iterator)
    except StopIteration:
        return out
    raise ValueError("flat values do not match shape")


def _pair_values(
    left_row: list[float],
    right_row: list[float],
    mul_left: int,
    mul_right: int,
    dim_left: int,
    dim_right: int,
    ir_left: str,
    ir_right: str,
    ir_out: str,
    coefficient: float,
) -> list[list[list[float]]]:
    cg = clebsch_gordan(ir_left, ir_right, ir_out)
    out_dim = len(cg[0][0])
    pair = [[[0.0 for _ in range(out_dim)] for _ in range(mul_right)] for _ in range(mul_left)]
    for u in range(mul_left):
        for v in range(mul_right):
            for c in range(out_dim):
                value = 0.0
                for a in range(dim_left):
                    for b in range(dim_right):
                        value += (
                            left_row[u * dim_left + a]
                            * right_row[v * dim_right + b]
                            * cg[a][b][c]
                        )
                pair[u][v][c] = coefficient * value
    return pair


def _apply_mode(pair: list[list[list[float]]], mode: str, path_shape: tuple[int, ...], weights: Any | None) -> list[list[float]]:
    mul_left = len(pair)
    mul_right = len(pair[0]) if pair else 0
    out_dim = len(pair[0][0]) if pair and pair[0] else 0

    if mode == "uvw":
        if weights is None:
            return [pair[u][v][:] for u in range(mul_left) for v in range(mul_right)]
        return [
            [
                sum(weights[w][u][v] * pair[u][v][c] for u in range(mul_left) for v in range(mul_right))
                for c in range(out_dim)
            ]
            for w in range(path_shape[2])
        ]

    if mode == "uvu":
        if weights is None:
            return [[sum(pair[u][v][c] for v in range(mul_right)) for c in range(out_dim)] for u in range(mul_left)]
        return [
            [sum(weights[u][v] * pair[u][v][c] for v in range(mul_right)) for c in range(out_dim)]
            for u in range(mul_left)
        ]

    if mode == "uvv":
        if weights is None:
            return [[sum(pair[u][v][c] for u in range(mul_left)) for c in range(out_dim)] for v in range(mul_right)]
        return [
            [sum(weights[u][v] * pair[u][v][c] for u in range(mul_left)) for c in range(out_dim)]
            for v in range(mul_right)
        ]

    if mode == "uuw":
        diag = [pair[u][u][:] for u in range(mul_left)]
        if weights is None:
            return [[sum(diag[u][c] for u in range(mul_left)) for c in range(out_dim)]]
        return [
            [sum(weights[u][w] * diag[u][c] for u in range(mul_left)) for c in range(out_dim)]
            for w in range(path_shape[1])
        ]

    if mode == "uuu":
        diag = [pair[u][u][:] for u in range(mul_left)]
        if weights is None:
            return diag
        return [
            [weights[u] * diag[u][c] for c in range(out_dim)]
            for u in range(mul_left)
        ]

    if mode == "uvuv":
        if weights is None:
            return [pair[u][v][:] for u in range(mul_left) for v in range(mul_right)]
        return [
            [weights[u][v] * pair[u][v][c] for c in range(out_dim)]
            for u in range(mul_left)
            for v in range(mul_right)
        ]

    upper = [pair[u][v][:] for u in range(mul_left) for v in range(u + 1, mul_right)]
    if mode == "uvu<v":
        if weights is None:
            return upper
        return [[weights[q] * upper[q][c] for c in range(out_dim)] for q in range(len(upper))]

    if mode == "u<vw":
        if weights is None:
            return [[sum(upper[q][c] for q in range(len(upper))) for c in range(out_dim)]]
        return [
            [sum(weights[q][w] * upper[q][c] for q in range(len(upper))) for c in range(out_dim)]
            for w in range(path_shape[1])
        ]

    raise ValueError(f"unsupported mode {mode!r}")


def _reference_tensor_product(tp: TensorProduct, left: IrrepsArray, right: IrrepsArray, *, weight: Any | None = None) -> list[list[float]]:
    batch_size = left.shape[0]
    left_chunks = _chunk_rows(left)
    right_chunks = _chunk_rows(right)
    weight_rows: dict[int, list[Any]] = {}
    if weight is not None:
        for meta in tp._weighted_instruction_meta:
            sliced = weight[..., meta.weight_slice]
            weight_rows[meta.instruction_index] = _reshape_weight_rows(sliced, meta.instruction.path_shape, batch_size, tp.shared_weights)

    outputs: list[list[float]] = []
    for batch_index in range(batch_size):
        grouped = [[] for _ in range(len(tp.irreps_out))]
        for instruction_index, inst in enumerate(tp.instructions):
            left_row = left_chunks[inst.input1_index][batch_index]
            right_row = right_chunks[inst.input2_index][batch_index]
            mul_left = tp.irreps_in1[inst.input1_index].mul
            mul_right = tp.irreps_in2[inst.input2_index].mul
            pair = _pair_values(
                left_row,
                right_row,
                mul_left,
                mul_right,
                inst.ir_in1.dim,
                inst.ir_in2.dim,
                str(inst.ir_in1),
                str(inst.ir_in2),
                str(inst.ir_out),
                inst.normalization.coefficient,
            )
            local_weight = weight_rows.get(instruction_index, [None] * batch_size)[batch_index] if weight is not None else None
            grouped[inst.output_index].extend(_apply_mode(pair, inst.mode, inst.path_shape, local_weight))
        flat = []
        for output_index, part in enumerate(tp.irreps_out):
            block = grouped[output_index]
            assert len(block) == part.mul
            for row in block:
                flat.extend(row)
        outputs.append(flat)
    return outputs


@pytest.mark.parametrize(
    ("mode", "irreps_in1", "irreps_in2", "irreps_out", "instructions", "weight", "shared_weights"),
    [
        ("uvw", "1o", "1o", "0e+1e+2e", [(0, 0, 0, "uvw", False), (0, 0, 1, "uvw", False), (0, 0, 2, "uvw", False)], None, True),
        ("uvu", "2x0e", "3x0e", "2x0e", [(0, 0, 0, "uvu", True)], [1.0, 0.5, -1.0, 2.0, 0.0, -0.5], True),
        ("uvv", "2x0e", "3x0e", "3x0e", [(0, 0, 0, "uvv", True)], [1.0, 0.0, -1.0, 0.5, 2.0, 0.0], True),
        ("uuw", "2x0e", "2x0e", "2x0e", [(0, 0, 0, "uuw", True)], [1.0, -1.0, 0.5, 2.0], True),
        ("uuu", "2x0e", "2x0e", "2x0e", [(0, 0, 0, "uuu", True)], [1.5, -0.25], True),
        ("uvuv", "2x0e", "3x0e", "6x0e", [(0, 0, 0, "uvuv", True)], [1.0, 0.5, -1.0, 2.0, 0.0, -0.5], True),
        ("uvu<v", "3x0e", "3x0e", "3x0e", [(0, 0, 0, "uvu<v", True)], [1.0, -1.0, 0.5], True),
        ("u<vw", "3x0e", "3x0e", "2x0e", [(0, 0, 0, "u<vw", True)], [1.0, 0.0, -1.0, 0.5, 2.0, 0.0], True),
    ],
)
def test_tensor_product_modes_match_independent_reference(
    mode: str,
    irreps_in1: str,
    irreps_in2: str,
    irreps_out: str,
    instructions: list[tuple[int, int, int, str, bool]],
    weight: list[float] | None,
    shared_weights: bool,
) -> None:
    left = IrrepsArray(irreps_in1, mlx_backend.asarray([[2.0] * Irreps(irreps_in1).dim, [3.0] * Irreps(irreps_in1).dim]))
    right = IrrepsArray(irreps_in2, mlx_backend.asarray([[5.0] * Irreps(irreps_in2).dim, [7.0] * Irreps(irreps_in2).dim]))
    tp = TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions, internal_weights=False, shared_weights=shared_weights)
    weight_array = None if weight is None else mlx_backend.asarray(weight)
    out = tp(left, right, weight=weight_array) if weight_array is not None else tp(left, right)
    ref = _reference_tensor_product(tp, left, right, weight=weight_array)
    assert _max_diff(out.array, ref) < 2e-5, mode


def test_tensor_product_unshared_weights_match_independent_reference() -> None:
    tp = TensorProduct(
        "2x0e",
        "3x0e",
        "2x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0], [5.0, 7.0]]))
    right = IrrepsArray("3x0e", mlx_backend.asarray([[11.0, 13.0, 17.0], [19.0, 23.0, 29.0]]))
    weights = mlx_backend.asarray([[1.0, 0.0, -1.0, 0.5, 0.5, 0.5], [0.0, 1.0, 1.0, 2.0, 0.0, -1.0]])
    out = tp(left, right, weight=weights)
    ref = _reference_tensor_product(tp, left, right, weight=weights)
    assert _max_diff(out.array, ref) < 2e-5


def test_full_tensor_product_matches_explicit_tensor_product() -> None:
    left = IrrepsArray("2x1o", mlx_backend.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]))
    right = IrrepsArray("0e+1e", mlx_backend.asarray([[7.0, 8.0, 9.0, 10.0]]))
    wrapper = FullTensorProduct("2x1o", "1x0e + 1x1e")
    explicit = TensorProduct(
        "2x1o",
        "0e+1e",
        "2x1o+2x0o+2x1o+2x2o",
        [
            (0, 0, 0, "uvuv", False),
            (0, 1, 1, "uvuv", False),
            (0, 1, 2, "uvuv", False),
            (0, 1, 3, "uvuv", False),
        ],
        internal_weights=False,
    )
    out = wrapper(left, right)
    ref = explicit(left, right)
    assert str(out.irreps) == "2x0o+4x1o+2x2o"
    regrouped = _regroup_flat_rows(ref.array.tolist(), "2x1o+2x0o+2x1o+2x2o", "2x0o+4x1o+2x2o")
    assert _max_diff(out.array, regrouped) < 2e-5


def test_fully_connected_tensor_product_matches_explicit_tensor_product() -> None:
    left = IrrepsArray("1o", mlx_backend.asarray([[1.0, 2.0, 3.0]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[4.0, 5.0, 6.0]]))
    weight = mlx_backend.asarray([0.5, -1.0, 2.0])
    wrapper = FullyConnectedTensorProduct("1o", "1o", "0e+1e+2e", internal_weights=False)
    explicit = TensorProduct(
        "1o",
        "1o",
        "0e+1e+2e",
        [
            (0, 0, 0, "uvw", True),
            (0, 0, 1, "uvw", True),
            (0, 0, 2, "uvw", True),
        ],
        internal_weights=False,
    )
    out = wrapper(left, right, weight=weight)
    ref = explicit(left, right, weight=weight)
    assert _max_diff(out.array, ref.array.tolist()) < 2e-5


def test_elementwise_tensor_product_matches_explicit_tensor_product() -> None:
    left = IrrepsArray("2x0e+1o", mlx_backend.asarray([[2.0, 3.0, 4.0, 5.0, 6.0]]))
    right = IrrepsArray("2x0e+1o", mlx_backend.asarray([[7.0, 11.0, 13.0, 17.0, 19.0]]))
    wrapper = ElementwiseTensorProduct("2x0e+1o", "2x0e+1o")
    explicit = TensorProduct(
        "2x0e+1o",
        "2x0e+1o",
        str(wrapper.irreps_out),
        [
            (0, 0, 0, "uuu", False),
            (1, 1, 1, "uuu", False),
            (1, 1, 2, "uuu", False),
            (1, 1, 3, "uuu", False),
        ],
        internal_weights=False,
    )
    out = wrapper(left, right)
    ref = explicit(left, right)
    assert _max_diff(out.array, ref.array.tolist()) < 2e-5


def test_tensor_square_matches_explicit_tensor_product_on_user_example() -> None:
    array = IrrepsArray("5x1e+2e", mlx_backend.asarray([[float(i) for i in range(1, 21)]]))
    wrapper = TensorSquare("5x1e + 2e")
    out = wrapper(array)
    assert str(out.irreps) == "16x0e+15x1e+21x2e+5x3e+4e"
    assert repr(wrapper) == "TensorSquare(5x1e+1x2e -> 16x0e+15x1e+21x2e+5x3e+1x4e | 58 paths | 0 weights)"
    assert out.shape == (1, Irreps("16x0e+15x1e+21x2e+5x3e+4e").dim)
