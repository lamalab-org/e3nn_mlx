"""Immutable O(3) irrep metadata and parsing."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable, Iterator

from .typing import Parity

_IRREP_RE = re.compile(r"^(?:(?P<mul>\d+)x)?(?P<l>\d+)(?P<parity>[eoy])$")


def _parse_parity_token(token: str, l: int) -> Parity:
    if token == "e":
        return 1
    if token == "o":
        return -1
    if token == "y":
        return 1 if l % 2 == 0 else -1
    raise ValueError(f"invalid parity token: {token!r}")


@dataclass(frozen=True, slots=True, order=True, init=False)
class Irrep:
    l: int
    p: Parity

    def __init__(self, l: int | str | tuple[int, int] | Irrep, p: int | None = None) -> None:
        if isinstance(l, Irrep):
            if p is not None:
                raise ValueError("p must not be provided when copying an Irrep")
            parsed_l, parsed_p = l.l, l.p
        elif isinstance(l, str):
            if p is not None:
                raise ValueError("p must not be provided with a string irrep")
            match = _IRREP_RE.match(l.replace(" ", ""))
            if match is None or match.group("mul") is not None:
                raise ValueError(f"invalid irrep spec: {l!r}")
            parsed_l = int(match.group("l"))
            parsed_p = _parse_parity_token(match.group("parity"), parsed_l)
        elif isinstance(l, tuple):
            if p is not None or len(l) != 2:
                raise ValueError("tuple irrep must be (l, p)")
            parsed_l, parsed_p = int(l[0]), int(l[1])
        else:
            if p is None:
                raise ValueError("parity p is required when l is an integer")
            parsed_l, parsed_p = int(l), int(p)
        object.__setattr__(self, "l", parsed_l)
        object.__setattr__(self, "p", parsed_p)
        self.__post_init__()

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
    def parse(cls, spec: str | tuple[int, int] | Irrep) -> Irrep:
        if isinstance(spec, cls):
            return spec
        return cls(spec)

    def is_scalar(self) -> bool:
        return self.l == 0 and self.p == 1

    def selection_rule(self, other: Irrep | str) -> tuple[Irrep, ...]:
        other = Irrep.parse(other)
        parity = self.p * other.p
        return tuple(Irrep(l_out, parity) for l_out in range(abs(self.l - other.l), self.l + other.l + 1))

    def __mul__(self, other: Irrep | str | int) -> tuple[Irrep, ...] | Irreps:
        if isinstance(other, int):
            if other < 0:
                raise ValueError("multiplicity must be non-negative")
            return Irreps(((other, self),))
        return self.selection_rule(other)

    def __rmul__(self, other: int) -> Irreps:
        result = self * other
        if not isinstance(result, Irreps):
            raise TypeError("Irrep can only be left-multiplied by an integer")
        return result

    def __add__(self, other: Irrep | str) -> Irreps:
        return Irreps(((1, self), (1, Irrep.parse(other))))

    def __str__(self) -> str:
        return f"{self.l}{self.parity_token}"

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self):
        return iter((self.l, self.p))

    @staticmethod
    def iterator(lmax: int | None = None) -> Iterator[Irrep]:
        if lmax is not None and lmax < 0:
            raise ValueError("lmax must be non-negative")
        l = 0
        while lmax is None or l <= lmax:
            yield Irrep(l, 1)
            yield Irrep(l, -1)
            l += 1


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
    def parse(cls, spec: str | Irrep | MulIrrep | tuple[int, Irrep | str]) -> MulIrrep:
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, Irrep):
            return cls(1, spec)
        if isinstance(spec, tuple):
            mul, ir = spec
            return cls(mul=int(mul), ir=Irrep.parse(ir))
        token = spec.replace(" ", "")
        match = _IRREP_RE.match(token)
        if match is None:
            raise ValueError(f"invalid mul irrep spec: {spec!r}")
        mul = int(match.group("mul") or 1)
        l = int(match.group("l"))
        return cls(mul=mul, ir=Irrep(l, _parse_parity_token(match.group("parity"), l)))

    def __str__(self) -> str:
        if self.mul == 1:
            return str(self.ir)
        return f"{self.mul}x{self.ir}"


@dataclass(frozen=True, slots=True)
class SortResult:
    irreps: Irreps
    p: tuple[int, ...]
    inv: tuple[int, ...]

    def __iter__(self):
        return iter((self.irreps, self.p, self.inv))


@dataclass(frozen=True, slots=True)
class Irreps:
    parts: tuple[MulIrrep, ...]

    def __init__(self, spec: str | Irrep | Irreps | Iterable[MulIrrep | Irrep | str | tuple[int, Irrep | str]] = ()) -> None:
        if isinstance(spec, Irreps):
            parts = spec.parts
        elif isinstance(spec, Irrep):
            parts = (MulIrrep(1, spec),)
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
        nonzero = [part.ir.l for part in self.parts if part.mul > 0]
        if not nonzero:
            return -1
        return max(nonzero)

    @property
    def ls(self) -> tuple[int, ...]:
        return tuple(part.ir.l for part in self.parts for _ in range(part.mul))

    def slices(self) -> tuple[slice, ...]:
        start = 0
        out: list[slice] = []
        for part in self.parts:
            stop = start + part.dim
            out.append(slice(start, stop))
            start = stop
        return tuple(out)

    def simplify(self) -> Irreps:
        simplified: list[MulIrrep] = []
        for part in self.parts:
            if part.mul == 0:
                continue
            if simplified and simplified[-1].ir == part.ir:
                previous = simplified[-1]
                simplified[-1] = MulIrrep(previous.mul + part.mul, part.ir)
            else:
                simplified.append(part)
        return Irreps(simplified)

    def remove_zero_multiplicities(self) -> Irreps:
        return Irreps(part for part in self.parts if part.mul > 0)

    def regroup(self) -> Irreps:
        return self.sort().irreps.simplify()

    def sort(self) -> SortResult:
        inv = tuple(sorted(range(len(self.parts)), key=lambda index: self.parts[index].ir))
        p_list = [0] * len(inv)
        for new_index, old_index in enumerate(inv):
            p_list[old_index] = new_index
        return SortResult(Irreps(self.parts[index] for index in inv), tuple(p_list), inv)

    def filter(
        self,
        keep: Irreps | str | Iterable[Irrep | str] | Callable[[MulIrrep], bool] | None = None,
        *,
        drop: Irreps | str | Iterable[Irrep | str] | Callable[[MulIrrep], bool] | None = None,
        lmax: int | None = None,
    ) -> Irreps:
        if keep is not None and drop is not None:
            raise ValueError("keep and drop are mutually exclusive")

        def predicate(value, default: bool) -> Callable[[MulIrrep], bool]:
            if value is None:
                return lambda _part: default
            if callable(value):
                return value
            if isinstance(value, (str, Irreps)):
                allowed = {part.ir for part in Irreps(value)}
            else:
                allowed = {Irrep.parse(ir) for ir in value}
            return lambda part: part.ir in allowed

        keep_predicate = predicate(keep, True)
        drop_predicate = predicate(drop, False)
        return Irreps(
            part
            for part in self.parts
            if keep_predicate(part) and not drop_predicate(part) and (lmax is None or part.ir.l <= lmax)
        )

    @staticmethod
    def spherical_harmonics(lmax: int, p: int = -1) -> Irreps:
        if lmax < 0:
            return Irreps()
        if p not in (-1, 1):
            raise ValueError("p must be -1 or 1")
        return Irreps((1, Irrep(l, p**l)) for l in range(lmax + 1))

    def extend(self, other: str | Irreps | Iterable[MulIrrep | str | tuple[int, Irrep | str]]) -> Irreps:
        return Irreps((*self.parts, *Irreps(other).parts))

    def count(self, irrep: str | Irrep) -> int:
        target = Irrep.parse(irrep)
        return sum(part.mul for part in self.parts if part.ir == target)

    def __iter__(self) -> Iterator[MulIrrep]:
        return iter(self.parts)

    def __len__(self) -> int:
        return len(self.parts)

    def __contains__(self, irrep: object) -> bool:
        try:
            target = Irrep.parse(irrep)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return any(part.ir == target and part.mul > 0 for part in self.parts)

    def __getitem__(self, item: int | slice) -> MulIrrep | Irreps:
        if isinstance(item, slice):
            return Irreps(self.parts[item])
        return self.parts[item]

    def __add__(self, other: str | Irreps | Iterable[MulIrrep | str | tuple[int, Irrep | str]]) -> Irreps:
        return self.extend(other)

    def __mul__(self, multiplicity: int) -> Irreps:
        if not isinstance(multiplicity, int):
            return NotImplemented
        if multiplicity < 0:
            raise ValueError("multiplicity must be non-negative")
        return Irreps(self.parts * multiplicity)

    def __rmul__(self, multiplicity: int) -> Irreps:
        return self * multiplicity

    @property
    def slice_by_mul(self) -> _MulIndexSlice:
        return _MulIndexSlice(self)

    def __str__(self) -> str:
        return "+".join(str(part) for part in self.parts)

    def __repr__(self) -> str:
        return f"Irreps('{self}')"


class _MulIndexSlice:
    """Multiplicity-indexed slicing helper matching upstream ``Irreps``."""

    def __init__(self, irreps: Irreps) -> None:
        self.irreps = irreps

    def __getitem__(self, item: slice) -> Irreps:
        if not isinstance(item, slice):
            raise TypeError("slice_by_mul only supports slices")
        if item.step not in (None, 1):
            raise ValueError("slice_by_mul does not support a step")
        total = self.irreps.num_irreps
        start, stop, _ = item.indices(total)
        if stop <= start:
            return Irreps()
        parts: list[MulIrrep] = []
        cursor = 0
        for part in self.irreps:
            part_start, part_stop = cursor, cursor + part.mul
            overlap = max(0, min(stop, part_stop) - max(start, part_start))
            if overlap:
                parts.append(MulIrrep(overlap, part.ir))
            cursor = part_stop
            if cursor >= stop:
                break
        return Irreps(parts)
