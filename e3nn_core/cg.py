"""Clebsch-Gordan bookkeeping metadata."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import functools
from math import factorial, sqrt

from .irreps import Irrep
from .wigner import change_basis_real_to_complex


@dataclass(frozen=True, slots=True)
class ClebschGordanKey:
    ir_in1: Irrep
    ir_in2: Irrep
    ir_out: Irrep

    @classmethod
    def from_irreps(cls, ir_in1: Irrep | str, ir_in2: Irrep | str, ir_out: Irrep | str) -> ClebschGordanKey:
        return cls(Irrep.parse(ir_in1), Irrep.parse(ir_in2), Irrep.parse(ir_out))


def _f(value: int | float) -> int:
    rounded = round(value)
    if rounded != value:
        raise ValueError(f"factorial argument must be integral, got {value}")
    return factorial(rounded)


@functools.lru_cache(maxsize=None)
def su2_clebsch_gordan(l1: int, l2: int, l3: int) -> tuple[tuple[tuple[float, ...], ...], ...]:
    dim1 = 2 * l1 + 1
    dim2 = 2 * l2 + 1
    dim3 = 2 * l3 + 1
    if l3 not in range(abs(l1 - l2), l1 + l2 + 1):
        return tuple(tuple(tuple(0.0 for _ in range(dim3)) for _ in range(dim2)) for _ in range(dim1))
    mat = [[[0.0 for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            m3 = m1 + m2
            if abs(m3) <= l3:
                mat[l1 + m1][l2 + m2][l3 + m3] = _su2_clebsch_gordan_coeff((l1, m1), (l2, m2), (l3, m3))
    return tuple(tuple(tuple(values) for values in row) for row in mat)


def _su2_clebsch_gordan_coeff(idx1: tuple[int, int], idx2: tuple[int, int], idx3: tuple[int, int]) -> float:
    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3
    if m3 != m1 + m2:
        return 0.0
    vmin = max(-j1 + j2 + m3, -j1 + m1, 0)
    vmax = min(j2 + j3 + m1, j3 - j1 + j2, j3 + m3)

    c = (
        (2.0 * j3 + 1.0)
        * Fraction(
            _f(j3 + j1 - j2) * _f(j3 - j1 + j2) * _f(j1 + j2 - j3) * _f(j3 + m3) * _f(j3 - m3),
            _f(j1 + j2 + j3 + 1) * _f(j1 - m1) * _f(j1 + m1) * _f(j2 - m2) * _f(j2 + m2),
        )
    ) ** 0.5
    acc = Fraction(0, 1)
    for v in range(vmin, vmax + 1):
        acc += ((-1) ** (v + j2 + m2)) * Fraction(
            _f(j2 + j3 + m1 - v) * _f(j1 - m1 + v),
            _f(v) * _f(j3 - j1 + j2 - v) * _f(j3 + m3 - v) * _f(v + j1 - j2 - m3),
        )
    return float(c * acc)


@functools.lru_cache(maxsize=None)
def clebsch_gordan(ir_in1: Irrep | str, ir_in2: Irrep | str, ir_out: Irrep | str) -> tuple[tuple[tuple[float, ...], ...], ...]:
    key = ClebschGordanKey.from_irreps(ir_in1, ir_in2, ir_out)
    return _so3_clebsch_gordan(key.ir_in1.l, key.ir_in2.l, key.ir_out.l)


@functools.lru_cache(maxsize=None)
def _so3_clebsch_gordan(l1: int, l2: int, l3: int) -> tuple[tuple[tuple[float, ...], ...], ...]:
    q1 = change_basis_real_to_complex(l1)
    q2 = change_basis_real_to_complex(l2)
    q3 = change_basis_real_to_complex(l3)
    su2 = su2_clebsch_gordan(l1, l2, l3)
    dim1 = 2 * l1 + 1
    dim2 = 2 * l2 + 1
    dim3 = 2 * l3 + 1
    out = [[[0j for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]
    for j in range(dim1):
        for l in range(dim2):
            for m in range(dim3):
                total = 0j
                for i in range(dim1):
                    for k in range(dim2):
                        for n in range(dim3):
                            total += q1[i][j] * q2[k][l] * q3[n][m].conjugate() * su2[i][k][n]
                out[j][l][m] = total

    real_out = [[[value.real for value in row] for row in plane] for plane in out]
    norm = sqrt(sum(value * value for plane in real_out for row in plane for value in row))
    if norm == 0.0:
        return tuple(tuple(tuple(0.0 for _ in range(dim3)) for _ in range(dim2)) for _ in range(dim1))
    return tuple(tuple(tuple(value / norm for value in row) for row in plane) for plane in real_out)
