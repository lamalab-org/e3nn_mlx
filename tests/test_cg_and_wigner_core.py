from __future__ import annotations

import math

from e3nn_core.cg import clebsch_gordan, su2_clebsch_gordan
from e3nn_core.wigner import change_basis_real_to_complex


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
