"""Wigner bookkeeping metadata."""

from __future__ import annotations

from dataclasses import dataclass

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
