"""MLX adaptation of the upstream NN Gate test."""

from __future__ import annotations

import pytest

tree_flatten = pytest.importorskip("mlx.utils").tree_flatten

import e3nn_mlx as e3nn
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.nn_gate import _Sortcut


def _array(irreps, values):
    return e3nn.IrrepsArray(irreps, values)


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value)))


@pytest.mark.mlx
def test_upstream_sortcut_and_gate_equivariance_compile_normalization() -> None:
    mx = mlx_backend._require()
    irreps_scalars = e3nn.Irreps("16x0o")
    irreps_gates = e3nn.Irreps("32x0o")
    irreps_gated = e3nn.Irreps("16x1e + 16x1o")

    # Match the standalone two-output _Sortcut construction in upstream's test.
    sortcut = _Sortcut(irreps_scalars, irreps_gates)
    values = mx.random.normal(shape=(8, sortcut.irreps_in.dim))
    outputs = sortcut(_array(sortcut.irreps_in, values))
    assert tuple(output.irreps for output in outputs) == sortcut.irreps_outs
    assert sum(output.irreps.dim for output in outputs) == sortcut.irreps_in.dim
    compiled_cut = mx.compile(
        lambda raw: tuple(output.array for output in sortcut(_array(sortcut.irreps_in, raw)))
    )
    compiled_outputs = compiled_cut(values)
    assert all(_max_abs(actual - expected.array) < 2e-6 for actual, expected in zip(compiled_outputs, outputs, strict=True))

    gate = e3nn.Gate(
        irreps_scalars,
        [mx.tanh],
        irreps_gates,
        [mx.tanh],
        irreps_gated,
    )
    assert gate.irreps_out == e3nn.Irreps("16x0o + 16x1o + 16x1e")
    samples = mx.random.normal(shape=(50_000, gate.irreps_in.dim))
    output = gate(_array(gate.irreps_in, samples))
    assert 0.93 < float(mx.mean(output.array**2)) < 1.07

    angles = e3nn.rand_angles()
    for inversion in (0, 1):
        d_in = e3nn.irreps_wigner_d(gate.irreps_in, *angles, k=inversion)
        d_out = e3nn.irreps_wigner_d(gate.irreps_out, *angles, k=inversion)
        actual = gate(_array(gate.irreps_in, samples[:16] @ mx.swapaxes(d_in, -1, -2))).array
        expected = output.array[:16] @ mx.swapaxes(d_out, -1, -2)
        assert _max_abs(actual - expected) < 2e-4

    compiled = mx.compile(lambda raw: gate(_array(gate.irreps_in, raw)).array)
    assert _max_abs(compiled(samples[:16]) - output.array[:16]) < 2e-6
    assert tree_flatten(gate.parameters()) == []


@pytest.mark.mlx
def test_upstream_gate_validation_and_empty_gates() -> None:
    mx = mlx_backend._require()
    with pytest.raises(ValueError, match="only scalar"):
        e3nn.Gate("1o", [None], "", [], "")
    with pytest.raises(ValueError, match="only scalar"):
        e3nn.Gate("", [], "1o", [None], "1o")
    with pytest.raises(ValueError, match="number of gate"):
        e3nn.Gate("0e", [mx.tanh], "2x0e", [mx.sigmoid], "1o")

    scalar_only = e3nn.Gate("2x0e", [mx.tanh], "", [], "")
    values = mx.random.normal(shape=(4, 2))
    output = scalar_only(_array(scalar_only.irreps_in, values))
    assert output.irreps == e3nn.Irreps("2x0e")
    assert output.shape == values.shape
