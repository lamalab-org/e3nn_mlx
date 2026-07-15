from __future__ import annotations

import math

from e3nn_core.cg import clebsch_gordan, su2_clebsch_gordan, wigner_3j
from e3nn_core.wigner import change_basis_real_to_complex, so3_generators


def test_change_basis_real_to_complex_is_unitary_for_l1() -> None:
    q = change_basis_real_to_complex(1)
    ident = [[sum(q[r][i] * q[c][i].conjugate() for i in range(3)) for c in range(3)] for r in range(3)]
    for r in range(3):
        for c in range(3):
            expected = 1.0 if r == c else 0.0
            assert abs(ident[r][c] - expected) < 1e-12


def test_su2_clebsch_gordan_has_expected_shape() -> None:
    cg = su2_clebsch_gordan(2, 1, 3)
    assert len(cg) == 5
    assert len(cg[0]) == 3
    assert len(cg[0][0]) == 7


def test_so3_clebsch_gordan_is_normalized() -> None:
    cg = clebsch_gordan("2e", "1o", "3o")
    norm = math.sqrt(sum(value * value for plane in cg for row in plane for value in row))
    assert abs(norm - 1.0) < 1e-12


def test_change_basis_is_unitary_through_l8() -> None:
    for l in range(9):
        q = change_basis_real_to_complex(l)
        dim = 2 * l + 1
        for row in range(dim):
            for column in range(dim):
                value = sum(q[index][row].conjugate() * q[index][column] for index in range(dim))
                assert abs(value - (1.0 if row == column else 0.0)) < 1e-12


def test_real_generators_are_antisymmetric() -> None:
    for l in range(7):
        for generator in so3_generators(l):
            for row in range(2 * l + 1):
                for column in range(2 * l + 1):
                    assert abs(generator[row][column] + generator[column][row]) < 1e-12


def test_wigner_3j_validates_triangle_and_is_normalized() -> None:
    coefficients = wigner_3j(2, 3, 4)
    norm = math.sqrt(sum(value * value for plane in coefficients for row in plane for value in row))
    assert abs(norm - 1.0) < 1e-12
    try:
        wigner_3j(1, 1, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid triangle was accepted")
