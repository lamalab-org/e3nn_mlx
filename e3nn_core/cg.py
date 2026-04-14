"""Clebsch-Gordan bookkeeping metadata."""

from __future__ import annotations

from dataclasses import dataclass

from .irreps import Irrep


@dataclass(frozen=True, slots=True)
class ClebschGordanKey:
    ir_in1: Irrep
    ir_in2: Irrep
    ir_out: Irrep

    @classmethod
    def from_irreps(cls, ir_in1: Irrep | str, ir_in2: Irrep | str, ir_out: Irrep | str) -> ClebschGordanKey:
        return cls(Irrep.parse(ir_in1), Irrep.parse(ir_in2), Irrep.parse(ir_out))
