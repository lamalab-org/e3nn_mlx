"""MLX/core adaptations of upstream e3nn irreps and Wigner tests."""

from __future__ import annotations

import numpy as np
import pytest

import e3nn_mlx as o3
from e3nn_mlx.backend import mlx_backend


def test_upstream_irrep_creation_properties_and_iteration() -> None:
    irrep = o3.Irrep("3e")
    assert o3.Irrep(3, 1) == irrep
    assert o3.Irrep(irrep) == irrep
    assert o3.Irrep("10o") == o3.Irrep(10, -1)
    assert o3.Irrep("1y") == o3.Irrep("1o")
    assert o3.Irrep(repr(irrep)) == irrep
    assert tuple(irrep) == (3, 1)
    assert irrep.dim == 7

    assert len(list(o3.Irrep.iterator(5))) == 12
    iterator = o3.Irrep.iterator()
    for index in range(100):
        value = next(iterator)
        assert value.l == index // 2
        assert value.p in (-1, 1)


def test_upstream_irreps_creation_arithmetic_and_properties() -> None:
    irrep = o3.Irrep("4o")
    assert o3.Irreps(irrep) == o3.Irreps("4o")
    assert o3.Irreps([(32, (4, -1))]) == o3.Irreps("32x4o")
    assert o3.Irreps(["1e", "2o"]) == o3.Irreps("1e + 2o")
    assert o3.Irreps([(16, "3e"), "1e", (256, (1, -1))]) == o3.Irreps("16x3e + 1e + 256x1o")

    assert 3 * o3.Irrep("6o") == o3.Irreps("3x6o")
    assert tuple(o3.Irrep("1o") * o3.Irrep("2e")) == tuple(o3.Irreps("1o + 2o + 3o").parts[i].ir for i in range(3))
    assert o3.Irrep("4o") + o3.Irrep("7e") == o3.Irreps("4o + 7e")
    irreps = o3.Irreps("2x2e + 4x1o")
    assert 2 * irreps == irreps * 2 == o3.Irreps("2x2e + 4x1o + 2x2e + 4x1o")

    values = o3.Irreps("4x1e + 6x2e + 12x2o")
    assert values.ls == tuple([1] * 4 + [2] * 18)
    assert values.lmax == 2
    assert values.num_irreps == 22
    assert o3.Irrep("2e") in values
    assert o3.Irrep("2o") in values
    assert o3.Irrep("3e") not in values


def test_upstream_irreps_indexing_empty_errors_and_slice_by_mul() -> None:
    empty = o3.Irreps()
    assert empty == o3.Irreps("") == o3.Irreps([])
    assert len(empty) == empty.dim == empty.num_irreps == 0
    with pytest.raises(ValueError, match="empty Irreps"):
        empty.lmax

    irreps = o3.Irreps("16x1e + 3e + 2e + 5o")
    assert irreps[0].mul == 16 and irreps[0].ir == o3.Irrep("1e")
    assert irreps[-1].ir == o3.Irrep("5o")
    assert irreps[2:] == o3.Irreps("2e + 5o")

    assert o3.Irreps("10x0e").slice_by_mul[1:4] == o3.Irreps("3x0e")
    assert o3.Irreps("10x0e + 10x1e").slice_by_mul[5:15] == o3.Irreps("5x0e + 5x1e")
    assert o3.Irreps("10x0e + 2e + 10x1e").slice_by_mul[5:15] == o3.Irreps("5x0e + 2e + 4x1e")

    for invalid in (lambda: o3.Irrep(-1), lambda: o3.Irrep(1, 2), lambda: o3.Irrep("-1e")):
        with pytest.raises(ValueError):
            invalid()
    for invalid in ("-1x1e", "1x-1e", "bla"):
        with pytest.raises(ValueError):
            o3.Irreps(invalid)


def test_upstream_wigner_3j_permutation_symmetries() -> None:
    coefficient = np.asarray(o3.wigner_3j(1, 2, 3))
    assert np.allclose(coefficient, np.asarray(o3.wigner_3j(1, 3, 2)).transpose(0, 2, 1))
    assert np.allclose(coefficient, np.asarray(o3.wigner_3j(2, 1, 3)).transpose(1, 0, 2))
    assert np.allclose(coefficient, np.asarray(o3.wigner_3j(3, 2, 1)).transpose(2, 1, 0))
    assert np.allclose(coefficient, np.asarray(o3.wigner_3j(3, 1, 2)).transpose(1, 2, 0))
    assert np.allclose(coefficient, np.asarray(o3.wigner_3j(2, 3, 1)).transpose(2, 0, 1))


@pytest.mark.mlx
@pytest.mark.parametrize("l1,l2,l3", [(1, 2, 3), (2, 3, 4), (3, 4, 5), (1, 1, 1), (1, 1, 0), (2, 2, 2)])
def test_upstream_wigner_3j_is_rotation_invariant(l1: int, l2: int, l3: int) -> None:
    mx = mlx_backend._require()
    angles = o3.rand_angles(10)
    coefficient = mx.array(o3.wigner_3j(l1, l2, l3), dtype=mx.float32)
    d1 = o3.wigner_d(l1, *angles)
    d2 = o3.wigner_d(l2, *angles)
    d3 = o3.wigner_d(l3, *angles)
    transformed = mx.einsum("ijk,zil,zjm,zkn->zlmn", coefficient, d1, d2, d3)
    assert float(mx.max(mx.abs(coefficient - transformed))) < 2e-4


@pytest.mark.mlx
@pytest.mark.parametrize("j", [0, 0.5, 1, 1.5, 2, 2.5])
def test_upstream_su2_generator_algebra(j: float) -> None:
    mx = mlx_backend._require()
    generators = mx.array(o3.su2_generators(j), dtype=mx.complex64)

    def commutator(left, right):
        return left @ right - right @ left

    assert float(mx.max(mx.abs(commutator(generators[0], generators[1]) - generators[2]))) < 2e-5
    assert float(mx.max(mx.abs(commutator(generators[1], generators[2]) - generators[0]))) < 2e-5


@pytest.mark.mlx
def test_upstream_wigner_l1_is_cartesian_rotation() -> None:
    angles = o3.rand_angles(10)
    assert float((o3.angles_to_matrix(*angles) - o3.wigner_d(1, *angles)).abs().max()) < 2e-6
