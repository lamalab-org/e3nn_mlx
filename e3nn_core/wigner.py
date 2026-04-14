"""Wigner bookkeeping metadata."""

from __future__ import annotations

from dataclasses import dataclass
import functools

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
