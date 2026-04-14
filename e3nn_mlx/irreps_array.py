"""MLX-backed IrrepsArray."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from e3nn_core.irreps import Irrep, Irreps


@dataclass(frozen=True, slots=True)
class IrrepsChunk:
    irrep: Irrep
    multiplicity: int
    slice: slice

    @property
    def width(self) -> int:
        return self.slice.stop - self.slice.start


@dataclass(frozen=True, slots=True)
class IrrepsArray:
    irreps: Irreps
    array: Any
    leading_shape: tuple[int, ...]
    chunks: tuple[IrrepsChunk, ...]

    def __init__(self, irreps: Irreps | str, array: Any) -> None:
        parsed = Irreps(irreps).simplify()
        shape = tuple(int(dim) for dim in array.shape)
        if not shape:
            raise ValueError("IrrepsArray requires at least one array dimension")
        if shape[-1] != parsed.dim:
            raise ValueError(f"last dimension {shape[-1]} does not match irreps dim {parsed.dim}")
        object.__setattr__(self, "irreps", parsed)
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "leading_shape", shape[:-1])
        object.__setattr__(self, "chunks", _build_chunks(parsed))

    @classmethod
    def from_chunks(cls, irreps: Irreps | str, chunks: list[Any] | tuple[Any, ...], *, backend: Any) -> IrrepsArray:
        parsed = Irreps(irreps).simplify()
        expected = _build_chunks(parsed)
        if len(chunks) != len(expected):
            raise ValueError(f"expected {len(expected)} chunks, got {len(chunks)}")
        widths = [chunk.shape[-1] for chunk in chunks]
        if widths != [meta.width for meta in expected]:
            raise ValueError(f"chunk widths {widths} do not match irreps widths {[meta.width for meta in expected]}")
        array = backend.concatenate(tuple(chunks), axis=-1)
        return cls(parsed, array)

    @property
    def dtype(self) -> Any:
        return getattr(self.array, "dtype", None)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.array.shape)

    def chunk_arrays(self) -> tuple[Any, ...]:
        return tuple(self.array[..., meta.slice] for meta in self.chunks)

    def chunk_dict(self) -> dict[str, Any]:
        return {str(meta.irrep): self.array[..., meta.slice] for meta in self.chunks}

    def with_array(self, array: Any) -> IrrepsArray:
        return IrrepsArray(self.irreps, array)

    def regroup(self) -> IrrepsArray:
        return IrrepsArray(self.irreps.regroup(), self.array)

    def simplify(self) -> IrrepsArray:
        return IrrepsArray(self.irreps.simplify(), self.array)


def _build_chunks(irreps: Irreps) -> tuple[IrrepsChunk, ...]:
    chunks: list[IrrepsChunk] = []
    start = 0
    for part in irreps:
        width = part.dim
        chunks.append(IrrepsChunk(irrep=part.ir, multiplicity=part.mul, slice=slice(start, start + width)))
        start += width
    return tuple(chunks)
