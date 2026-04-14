"""Immutable O(3) irrep metadata and parsing."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Iterator

from .typing import Parity

_IRREP_RE = re.compile(r"^(?:(?P<mul>\d+)x)?(?P<l>\d+)(?P<parity>[eo])$")


def _parse_parity_token(token: str) -> Parity:
    if token == "e":
        return 1
    if token == "o":
        return -1
    raise ValueError(f"invalid parity token: {token!r}")


@dataclass(frozen=True, slots=True, order=True)
class Irrep:
    l: int
    p: Parity

    def __post_init__(self) -> None:
        if self.l < 0:
            raise ValueError("l must be >= 0")
        if self.p not in (-1, 1):
            raise ValueError("parity must be -1 or 1")

    @property
    def dim(self) -> int:
        return 2 * self.l + 1

    @property
    def parity_token(self) -> str:
        return "e" if self.p == 1 else "o"

    @classmethod
    def parse(cls, spec: str | Irrep) -> Irrep:
        if isinstance(spec, cls):
            return spec
        match = _IRREP_RE.match(spec.replace(" ", ""))
        if match is None or match.group("mul") is not None:
            raise ValueError(f"invalid irrep spec: {spec!r}")
        return cls(l=int(match.group("l")), p=_parse_parity_token(match.group("parity")))

    def selection_rule(self, other: Irrep) -> tuple[Irrep, ...]:
        other = Irrep.parse(other)
        parity = self.p * other.p
        return tuple(Irrep(l=l_out, p=parity) for l_out in range(abs(self.l - other.l), self.l + other.l + 1))

    def __mul__(self, other: Irrep) -> tuple[Irrep, ...]:
        return self.selection_rule(other)

    def __str__(self) -> str:
        return f"{self.l}{self.parity_token}"


@dataclass(frozen=True, slots=True, order=True)
class MulIrrep:
    mul: int
    ir: Irrep

    def __post_init__(self) -> None:
        if self.mul < 0:
            raise ValueError("mul must be >= 0")

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    @classmethod
    def parse(cls, spec: str | MulIrrep | tuple[int, Irrep | str]) -> MulIrrep:
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, tuple):
            mul, ir = spec
            return cls(mul=int(mul), ir=Irrep.parse(ir))
        token = spec.replace(" ", "")
        match = _IRREP_RE.match(token)
        if match is None:
            raise ValueError(f"invalid mul irrep spec: {spec!r}")
        mul = int(match.group("mul") or 1)
        return cls(mul=mul, ir=Irrep(l=int(match.group("l")), p=_parse_parity_token(match.group("parity"))))

    def __str__(self) -> str:
        if self.mul == 1:
            return str(self.ir)
        return f"{self.mul}x{self.ir}"


@dataclass(frozen=True, slots=True)
class Irreps:
    parts: tuple[MulIrrep, ...]

    def __init__(self, spec: str | Irreps | Iterable[MulIrrep | str | tuple[int, Irrep | str]] = ()) -> None:
        if isinstance(spec, Irreps):
            parts = spec.parts
        elif isinstance(spec, str):
            token = spec.strip()
            if not token:
                parts = ()
            else:
                parts = tuple(MulIrrep.parse(piece.strip()) for piece in token.split("+"))
        else:
            parts = tuple(MulIrrep.parse(part) for part in spec)
        object.__setattr__(self, "parts", parts)

    @property
    def dim(self) -> int:
        return sum(part.dim for part in self.parts)

    @property
    def num_irreps(self) -> int:
        return sum(part.mul for part in self.parts)

    @property
    def lmax(self) -> int:
        if not self.parts:
            return -1
        return max(part.ir.l for part in self.parts)

    def slices(self) -> tuple[slice, ...]:
        start = 0
        out: list[slice] = []
        for part in self.parts:
            stop = start + part.dim
            out.append(slice(start, stop))
            start = stop
        return tuple(out)

    def simplify(self) -> Irreps:
        return Irreps(part for part in self.parts if part.mul > 0)

    def regroup(self) -> Irreps:
        grouped: list[MulIrrep] = []
        for part in self.parts:
            if part.mul == 0:
                continue
            if grouped and grouped[-1].ir == part.ir:
                prev = grouped[-1]
                grouped[-1] = MulIrrep(prev.mul + part.mul, prev.ir)
            else:
                grouped.append(part)
        return Irreps(grouped)

    def sort(self) -> Irreps:
        return Irreps(sorted(self.parts, key=lambda part: (part.ir.l, part.ir.p, part.mul))).regroup()

    def extend(self, other: str | Irreps | Iterable[MulIrrep | str | tuple[int, Irrep | str]]) -> Irreps:
        return Irreps((*self.parts, *Irreps(other).parts))

    def count(self, irrep: str | Irrep) -> int:
        target = Irrep.parse(irrep)
        return sum(part.mul for part in self.parts if part.ir == target)

    def __iter__(self) -> Iterator[MulIrrep]:
        return iter(self.parts)

    def __len__(self) -> int:
        return len(self.parts)

    def __getitem__(self, item: int | slice) -> MulIrrep | Irreps:
        if isinstance(item, slice):
            return Irreps(self.parts[item])
        return self.parts[item]

    def __add__(self, other: str | Irreps | Iterable[MulIrrep | str | tuple[int, Irrep | str]]) -> Irreps:
        return self.extend(other)

    def __str__(self) -> str:
        return "+".join(str(part) for part in self.parts)

    def __repr__(self) -> str:
        return f"Irreps('{self}')"
