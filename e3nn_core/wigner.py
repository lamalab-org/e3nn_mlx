"""Wigner bookkeeping metadata."""

from __future__ import annotations

from dataclasses import dataclass
import functools
from math import sqrt

from .irreps import Irrep


@dataclass(frozen=True, slots=True)
class Wigner3jKey:
    l1: int
    l2: int
    l3: int


@dataclass(frozen=True, slots=True)
class WignerDKey:
    l: int
    convention: str = "zyz"

    @property
    def irrep(self) -> Irrep:
        return Irrep(self.l, 1)


@functools.lru_cache(maxsize=None)
def change_basis_real_to_complex(l: int) -> tuple[tuple[complex, ...], ...]:
    if not isinstance(l, int) or l < 0:
        raise ValueError("l must be a non-negative integer")
    dim = 2 * l + 1
    q = [[0j for _ in range(dim)] for _ in range(dim)]
    s = 2.0**-0.5
    for m in range(-l, 0):
        q[l + m][l + abs(m)] = s
        q[l + m][l - abs(m)] = -1j * s
    q[l][l] = 1.0 + 0j
    for m in range(1, l + 1):
        q[l + m][l + abs(m)] = ((-1) ** m) * s
        q[l + m][l - abs(m)] = 1j * ((-1) ** m) * s
    phase = (-1j) ** l
    return tuple(tuple(phase * value for value in row) for row in q)


@functools.lru_cache(maxsize=None)
def su2_generators(l: int) -> tuple[tuple[tuple[complex, ...], ...], ...]:
    """Return the three anti-Hermitian SU(2) generators used by e3nn."""

    if not isinstance(l, int) or l < 0:
        raise ValueError("l must be a non-negative integer")
    dim = 2 * l + 1
    raising = [[0j for _ in range(dim)] for _ in range(dim)]
    lowering = [[0j for _ in range(dim)] for _ in range(dim)]
    for index, m in enumerate(range(-l, l)):
        raising[index + 1][index] = -sqrt(l * (l + 1) - m * (m + 1))
    for index, m in enumerate(range(-l + 1, l + 1)):
        lowering[index][index + 1] = sqrt(l * (l + 1) - m * (m - 1))

    generators = [[[0j for _ in range(dim)] for _ in range(dim)] for _ in range(3)]
    for row in range(dim):
        for column in range(dim):
            generators[0][row][column] = 0.5 * (raising[row][column] + lowering[row][column])
            generators[2][row][column] = -0.5j * (raising[row][column] - lowering[row][column])
    for index, m in enumerate(range(-l, l + 1)):
        generators[1][index][index] = 1j * m
    return tuple(tuple(tuple(row) for row in generator) for generator in generators)


@functools.lru_cache(maxsize=None)
def so3_generators(l: int) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Return e3nn's real anti-symmetric SO(3) generators."""

    q = change_basis_real_to_complex(l)
    x = su2_generators(l)
    dim = 2 * l + 1
    out = [[[0.0 for _ in range(dim)] for _ in range(dim)] for _ in range(3)]
    for axis in range(3):
        for row in range(dim):
            for column in range(dim):
                value = 0j
                for left in range(dim):
                    for right in range(dim):
                        value += q[left][row].conjugate() * x[axis][left][right] * q[right][column]
                if abs(value.imag) > 1e-10:
                    raise ArithmeticError("real-basis generator has a non-negligible imaginary component")
                out[axis][row][column] = value.real
    return tuple(tuple(tuple(row) for row in generator) for generator in out)
