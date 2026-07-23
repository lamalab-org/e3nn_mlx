"""MLX adaptations of upstream e3nn reduced-tensor-product tests."""

from __future__ import annotations

import copy

import pytest

import e3nn_mlx as o3
from e3nn_mlx.backend import mlx_backend


def _array(irreps, values):
    return o3.IrrepsArray(irreps, values)


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value)))


def _assert_equivariant(module, inputs) -> None:
    mx = mlx_backend._require()
    angles = o3.rand_angles()
    rotated_inputs = []
    for irreps, values in zip(module.irreps_in, inputs, strict=True):
        representation = o3.irreps_wigner_d(irreps, *angles)
        rotated_inputs.append(_array(irreps, values @ mx.swapaxes(representation, -1, -2)))
    actual = module(*rotated_inputs).array
    output = module(*(_array(irreps, values) for irreps, values in zip(module.irreps_in, inputs, strict=True))).array
    representation = o3.irreps_wigner_d(module.irreps_out, *angles)
    expected = output @ mx.swapaxes(representation, -1, -2)
    assert _max_abs(actual - expected) < 8e-4


@pytest.mark.mlx
def test_upstream_reduced_tensor_antisymmetric_matrix() -> None:
    mx = mlx_backend._require()
    module = o3.ReducedTensorProducts("ij=-ji", i="5x0e + 1e")
    assert module.change_of_basis.shape == (28, 8, 8)
    assert module.irreps_out.dim == 28
    assert _max_abs(module.change_of_basis + mx.swapaxes(module.change_of_basis, -1, -2)) < 2e-6

    inputs = [mx.random.normal(shape=(4, 8)), mx.random.normal(shape=(4, 8))]
    expected = mx.einsum("xij,zi,zj->zx", module.change_of_basis, *inputs)
    actual = module(_array(module.irreps_in[0], inputs[0]), _array(module.irreps_in[1], inputs[1])).array
    assert _max_abs(actual - expected) < 2e-6
    _assert_equivariant(module, inputs)

    duplicate = copy.deepcopy(module)
    assert _max_abs(duplicate.change_of_basis - module.change_of_basis) == 0.0
    assert _max_abs(duplicate(_array(module.irreps_in[0], inputs[0]), _array(module.irreps_in[1], inputs[1])).array - actual) == 0.0


@pytest.mark.mlx
@pytest.mark.parametrize("irrep,expected_irreps", [("1e", "0e"), ("2e", "1e + 3e")])
def test_upstream_reduced_tensor_fully_antisymmetric_rank_three(irrep: str, expected_irreps: str) -> None:
    mx = mlx_backend._require()
    module = o3.ReducedTensorProducts("ijk=-ikj=-jik", i=irrep)
    assert module.irreps_out == o3.Irreps(expected_irreps)
    basis = module.change_of_basis
    assert _max_abs(basis + mx.transpose(basis, (0, 1, 3, 2))) < 3e-6
    assert _max_abs(basis + mx.transpose(basis, (0, 2, 1, 3))) < 3e-6

    dim = o3.Irreps(irrep).dim
    inputs = [mx.random.normal(shape=(3, dim)) for _ in range(3)]
    expected = mx.einsum("xijk,zi,zj,zk->zx", basis, *inputs)
    actual = module(*(_array(irrep, values) for values in inputs)).array
    assert _max_abs(actual - expected) < 3e-6
    _assert_equivariant(module, inputs)


@pytest.mark.mlx
@pytest.mark.parametrize("irrep", ["1e", "1o"])
def test_upstream_reduced_tensor_elasticity_symmetries_and_parity(irrep: str) -> None:
    mx = mlx_backend._require()
    module = o3.ReducedTensorProducts("ijkl=jikl=klij", i=irrep)
    assert module.irreps_out.dim == 21
    assert all(part.ir.p == 1 for part in module.irreps_out)
    basis = module.change_of_basis
    assert _max_abs(basis - mx.transpose(basis, (0, 2, 1, 3, 4))) < 4e-6
    assert _max_abs(basis - mx.transpose(basis, (0, 1, 2, 4, 3))) < 4e-6
    assert _max_abs(basis - mx.transpose(basis, (0, 3, 4, 1, 2))) < 4e-6

    inputs = [mx.random.normal(shape=(2, 3)) for _ in range(4)]
    expected = mx.einsum("xijkl,zi,zj,zk,zl->zx", basis, *inputs)
    actual = module(*(_array(irrep, values) for values in inputs)).array
    assert _max_abs(actual - expected) < 4e-6
    _assert_equivariant(module, inputs)


@pytest.mark.mlx
def test_upstream_reduced_tensor_change_of_basis_is_orthonormal_and_compilable() -> None:
    mx = mlx_backend._require()
    module = o3.ReducedTensorProducts("ijkl=jikl=klij", i="1e")
    flat = module.change_of_basis.reshape(module.irreps_out.dim, -1)
    assert _max_abs(flat @ mx.swapaxes(flat, -1, -2) - mx.eye(module.irreps_out.dim)) < 4e-6

    values = [mx.random.normal(shape=(5, 3)) for _ in range(4)]
    compiled = mx.compile(
        lambda a, b, c, d: module(_array("1e", a), _array("1e", b), _array("1e", c), _array("1e", d)).array
    )
    expected = module(*(_array("1e", value) for value in values)).array
    assert _max_abs(compiled(*values) - expected) < 3e-6


@pytest.mark.mlx
def test_upstream_reduced_tensor_supports_intermediate_and_output_filters() -> None:
    mx = mlx_backend._require()
    irreps = o3.Irreps.spherical_harmonics(4)
    allowed_mid = list(o3.Irrep.iterator(lmax=4))
    allowed_out = list(o3.Irrep.iterator(lmax=0))
    module = o3.ReducedTensorProducts(
        "ijk=jik=ikj",
        i=irreps,
        filter_ir_mid=allowed_mid,
        filter_ir_out=allowed_out,
    )

    assert all(part.ir.l == 0 for part in module.irreps_out)
    assert module.change_of_basis.shape[1:] == (irreps.dim,) * 3
    inputs = [mx.random.normal(shape=(2, irreps.dim)) for _ in range(3)]
    expected = mx.einsum("xijk,zi,zj,zk->zx", module.change_of_basis, *inputs)
    actual = module(
        *(_array(irreps, values) for values in inputs)
    ).array
    assert _max_abs(actual - expected) < 3e-6
    _assert_equivariant(module, inputs)
