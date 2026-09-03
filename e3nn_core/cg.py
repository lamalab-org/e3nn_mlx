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


def _magnetic_numbers(j: float) -> tuple[float, ...]:
    """Return -j, -j+1, ..., j, exact for integer and half-integer j alike."""

    doubled = round(2 * j)
    if doubled % 2 == 0:
        # Integer j: keep exact ints so the integer path is unchanged.
        whole = doubled // 2
        return tuple(range(-whole, whole + 1))
    return tuple(half / 2 for half in range(-doubled, doubled + 1, 2))


@functools.lru_cache(maxsize=None)
def su2_clebsch_gordan(l1: int | float, l2: int | float, l3: int | float) -> tuple[tuple[tuple[float, ...], ...], ...]:
    for j in (l1, l2, l3):
        if not isinstance(j, (int, float)) or j < 0 or not float(2 * j).is_integer():
            raise ValueError("angular momenta must be non-negative integers or half-integers")
    dim1 = round(2 * l1) + 1
    dim2 = round(2 * l2) + 1
    dim3 = round(2 * l3) + 1
    # Triangle rule in doubled units, so half-integer couplings are handled too.
    if round(2 * l3) not in range(round(2 * abs(l1 - l2)), round(2 * (l1 + l2)) + 1, 2):
        return tuple(tuple(tuple(0.0 for _ in range(dim3)) for _ in range(dim2)) for _ in range(dim1))
    mat = [[[0.0 for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]
    for m1 in _magnetic_numbers(l1):
        for m2 in _magnetic_numbers(l2):
            m3 = m1 + m2
            if abs(m3) <= l3:
                mat[round(l1 + m1)][round(l2 + m2)][round(l3 + m3)] = _su2_clebsch_gordan_coeff(
                    (l1, m1), (l2, m2), (l3, m3)
                )
    return tuple(tuple(tuple(values) for values in row) for row in mat)


def _su2_clebsch_gordan_coeff(
    idx1: tuple[float, float], idx2: tuple[float, float], idx3: tuple[float, float]
) -> float:
    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3
    if m3 != m1 + m2:
        return 0.0
    vmin = round(max(-j1 + j2 + m3, -j1 + m1, 0))
    vmax = round(min(j2 + j3 + m1, j3 - j1 + j2, j3 + m3))

    c = (
        (2.0 * j3 + 1.0)
        * Fraction(
            _f(j3 + j1 - j2) * _f(j3 - j1 + j2) * _f(j1 + j2 - j3) * _f(j3 + m3) * _f(j3 - m3),
            _f(j1 + j2 + j3 + 1) * _f(j1 - m1) * _f(j1 + m1) * _f(j2 - m2) * _f(j2 + m2),
        )
    ) ** 0.5
    acc = Fraction(0, 1)
    for v in range(vmin, vmax + 1):
        acc += ((-1) ** round(v + j2 + m2)) * Fraction(
            _f(j2 + j3 + m1 - v) * _f(j1 - m1 + v),
            _f(v) * _f(j3 - j1 + j2 - v) * _f(j3 + m3 - v) * _f(v + j1 - j2 - m3),
        )
    return float(c * acc)


@functools.lru_cache(maxsize=None)
def clebsch_gordan(ir_in1: Irrep | str, ir_in2: Irrep | str, ir_out: Irrep | str) -> tuple[tuple[tuple[float, ...], ...], ...]:
    key = ClebschGordanKey.from_irreps(ir_in1, ir_in2, ir_out)
    # These arguments carry a parity, unlike wigner_3j's bare degrees, so an
    # O(3)-forbidden coupling is a caller error rather than a zero block.
    if key.ir_in1.p * key.ir_in2.p != key.ir_out.p:
        raise ValueError(
            f"parity mismatch: {key.ir_in1} x {key.ir_in2} cannot couple to {key.ir_out}"
        )
    return _so3_clebsch_gordan(key.ir_in1.l, key.ir_in2.l, key.ir_out.l)


@functools.lru_cache(maxsize=None)
def wigner_3j(l1: int, l2: int, l3: int) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Return e3nn-normalized real Wigner 3j coefficients."""

    if not all(isinstance(l, int) and l >= 0 for l in (l1, l2, l3)):
        raise ValueError("angular momenta must be non-negative integers")
    if not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError("angular momenta do not satisfy the triangle inequality")
    return _so3_clebsch_gordan(l1, l2, l3)


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
