"""Symmetry-reduced tensor products in the canonical e3nn basis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from string import ascii_lowercase
from typing import Any

from e3nn_core.cg import clebsch_gordan
from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from .compat import mlx_module_base, require_mlx
from .irreps_array import IrrepsArray


@dataclass(frozen=True, slots=True)
class _CoupledChannel:
    irrep: Irrep
    basis: Any


def _parse_formula(formula: str) -> tuple[str, tuple[tuple[int, tuple[int, ...]], ...]]:
    terms = formula.replace(" ", "").split("=")
    if not terms or not terms[0] or len(set(terms[0])) != len(terms[0]):
        raise ValueError("the first formula term must contain distinct tensor indices")
    base = terms[0]
    constraints = []
    for raw_term in terms[1:]:
        sign = -1 if raw_term.startswith("-") else 1
        term = raw_term[1:] if raw_term[:1] in ("-", "+") else raw_term
        if sorted(term) != sorted(base):
            raise ValueError(f"formula term {raw_term!r} is not a permutation of {base!r}")
        permutation = tuple(term.index(label) for label in base)
        constraints.append((sign, permutation))
    return base, tuple(constraints)


def _factor_irreps(base: str, supplied: dict[str, Irreps | str]) -> tuple[Irreps, ...]:
    if not supplied:
        raise ValueError("at least one index irrep must be provided")
    unknown = set(supplied) - set(base)
    if unknown:
        raise ValueError(f"irreps were supplied for indices not present in the formula: {sorted(unknown)}")
    if len(supplied) == 1:
        only = Irreps(next(iter(supplied.values()))).remove_zero_multiplicities()
        return tuple(only for _ in base)
    missing = [label for label in base if label not in supplied]
    if missing:
        raise ValueError(f"missing irreps for tensor indices {missing}")
    return tuple(Irreps(supplied[label]).remove_zero_multiplicities() for label in base)


def _irrep_embeddings(irreps: Irreps, np) -> list[tuple[Irrep, Any]]:
    embeddings = []
    cursor = 0
    for part in irreps:
        for copy_index in range(part.mul):
            embedding = np.zeros((part.ir.dim, irreps.dim), dtype=np.float64)
            start = cursor + copy_index * part.ir.dim
            embedding[:, start : start + part.ir.dim] = np.eye(part.ir.dim)
            embeddings.append((part.ir, embedding))
        cursor += part.dim
    return embeddings


def _couple_channels(factors: tuple[Irreps, ...], np) -> list[_CoupledChannel]:
    channels = [_CoupledChannel(irrep, embedding) for irrep, embedding in _irrep_embeddings(factors[0], np)]
    for factor in factors[1:]:
        next_channels = []
        for channel in channels:
            for right_irrep, right_basis in _irrep_embeddings(factor, np):
                for output_irrep in channel.irrep * right_irrep:
                    coefficient = np.asarray(
                        clebsch_gordan(channel.irrep, right_irrep, output_irrep), dtype=np.float64
                    )
                    basis = sqrt(output_irrep.dim) * np.einsum(
                        "a...,bj,abc->c...j", channel.basis, right_basis, coefficient, optimize=True
                    )
                    next_channels.append(_CoupledChannel(output_irrep, basis))
        channels = next_channels
    return channels


def _nullspace(matrix, np, tolerance: float = 1e-9):
    if matrix.shape[0] == 0:
        return np.eye(matrix.shape[1], dtype=np.float64)
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(singular_values > tolerance))
    return right[rank:].T


@lru_cache(maxsize=None)
def _build_change_of_basis(
    formula: str,
    factor_specs: tuple[str, ...],
) -> tuple[Irreps, Any]:
    import numpy as np

    base, constraints = _parse_formula(formula)
    factors = tuple(Irreps(spec) for spec in factor_specs)
    channels = _couple_channels(factors, np)
    grouped: dict[Irrep, list[Any]] = {}
    for channel in channels:
        grouped.setdefault(channel.irrep, []).append(channel.basis)

    output_parts = []
    output_bases = []
    tensor_axes = tuple(range(len(base)))
    for irrep in sorted(grouped):
        basis = np.stack(grouped[irrep], axis=0)
        channel_count = basis.shape[0]
        equations = []
        for sign, permutation in constraints:
            permuted = np.transpose(basis, (0, 1, *(axis + 2 for axis in permutation)))
            action = np.einsum("tm...,sm...->ts", basis, permuted, optimize=True)
            action /= irrep.dim
            equations.append(action - sign * np.eye(channel_count))
        matrix = np.concatenate(equations, axis=0) if equations else np.zeros((0, channel_count))
        invariant_channels = _nullspace(matrix, np)
        if invariant_channels.shape[1] == 0:
            continue
        reduced = np.einsum("ck,cm...->km...", invariant_channels, basis, optimize=True)
        output_parts.append(MulIrrep(invariant_channels.shape[1], irrep))
        output_bases.append(reduced.reshape(invariant_channels.shape[1] * irrep.dim, *reduced.shape[2:]))

    irreps_out = Irreps(output_parts)
    if output_bases:
        change_of_basis = np.concatenate(output_bases, axis=0)
    else:
        change_of_basis = np.zeros((0, *(factor.dim for factor in factors)), dtype=np.float64)
    return irreps_out, change_of_basis


class ReducedTensorProducts(mlx_module_base()):
    """Reduce a tensor product according to a permutation-symmetry formula."""

    def __init__(self, formula: str, **irreps: Irreps | str) -> None:
        super().__init__()
        base, _ = _parse_formula(formula)
        self.formula = formula
        self._labels = base
        self.irreps_in = _factor_irreps(base, irreps)
        self.irreps_out, data = _build_change_of_basis(formula, tuple(str(value) for value in self.irreps_in))
        mx, _ = require_mlx()
        self._change_of_basis = mx.array(data, dtype=mx.float32)

    @property
    def change_of_basis(self):
        return self._change_of_basis

    def __call__(self, *inputs: IrrepsArray) -> IrrepsArray:
        if len(inputs) != len(self.irreps_in):
            raise ValueError(f"expected {len(self.irreps_in)} inputs, got {len(inputs)}")
        for index, (array, expected) in enumerate(zip(inputs, self.irreps_in, strict=True)):
            if array.irreps != expected:
                raise ValueError(f"input {index} irreps {array.irreps} do not match {expected}")
        if len(inputs) > len(ascii_lowercase) - 1:
            raise ValueError("too many tensor indices")
        mx, _ = require_mlx()
        indices = ascii_lowercase[: len(inputs)]
        equation = f"x{indices}," + ",".join(f"...{index}" for index in indices) + "->...x"
        output = mx.einsum(equation, self._change_of_basis, *(array.array for array in inputs))
        return IrrepsArray(self.irreps_out, output)
