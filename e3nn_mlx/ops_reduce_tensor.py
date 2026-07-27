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


def _couple_channels(
    factors: tuple[Irreps, ...],
    np,
    filter_ir_mid: frozenset[Irrep] | None = None,
) -> list[_CoupledChannel]:
    channels = [_CoupledChannel(irrep, embedding) for irrep, embedding in _irrep_embeddings(factors[0], np)]
    for factor in factors[1:]:
        next_channels = []
        for channel in channels:
            for right_irrep, right_basis in _irrep_embeddings(factor, np):
                for output_irrep in channel.irrep * right_irrep:
                    if filter_ir_mid is not None and output_irrep not in filter_ir_mid:
                        continue
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
    filter_ir_mid_specs: tuple[str, ...] | None = None,
    filter_ir_out_specs: tuple[str, ...] | None = None,
) -> tuple[Irreps, Any]:
    import numpy as np

    base, constraints = _parse_formula(formula)
    factors = tuple(Irreps(spec) for spec in factor_specs)
    filter_ir_mid = (
        None
        if filter_ir_mid_specs is None
        else frozenset(Irrep.parse(spec) for spec in filter_ir_mid_specs)
    )
    filter_ir_out = (
        None
        if filter_ir_out_specs is None
        else frozenset(Irrep.parse(spec) for spec in filter_ir_out_specs)
    )
    channels = _couple_channels(factors, np, filter_ir_mid)
    grouped: dict[Irrep, list[Any]] = {}
    for channel in channels:
        if filter_ir_out is not None and channel.irrep not in filter_ir_out:
            continue
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
    """Reduce a tensor product according to permutation symmetries.

    Parameters
    ----------
    formula
        Index formula describing the tensor symmetry. For example, ``"ij=ji"``
        selects symmetric rank-two tensors and ``"ijk=jik=ikj"`` selects fully
        symmetric rank-three tensors.
    filter_ir_out
        Optional iterable of allowed final irreps. Channels with other output
        irreps are omitted from :attr:`irreps_out` and
        :attr:`change_of_basis`.
    filter_ir_mid
        Optional iterable of irreps allowed at every sequential coupling stage,
        including the final coupling. This prunes contraction paths before the
        permutation-symmetry reduction and can substantially reduce basis
        construction work.
    **irreps
        Irreps associated with formula indices. If only one index is supplied,
        its irreps are used for every index in the formula.

    Notes
    -----
    ``filter_ir_mid`` is a path constraint, not merely an output slice. It can
    change which multiplicity channels survive the symmetry reduction.

    Examples
    --------
    Construct scalar bispectrum contractions from spherical coefficients:

    >>> from e3nn_mlx import o3
    >>> lmax = 4
    >>> bispectrum = o3.ReducedTensorProducts(
    ...     "ijk=jik=ikj",
    ...     i=o3.Irreps.spherical_harmonics(lmax),
    ...     filter_ir_mid=list(o3.Irrep.iterator(lmax=lmax)),
    ...     filter_ir_out=list(o3.Irrep.iterator(lmax=0)),
    ... )
    >>> all(part.ir.l == 0 for part in bispectrum.irreps_out)
    True
    """

    def __init__(
        self,
        formula: str,
        filter_ir_out: list[Irrep | str] | None = None,
        filter_ir_mid: list[Irrep | str] | None = None,
        **irreps: Irreps | str,
    ) -> None:
        super().__init__()
        base, _ = _parse_formula(formula)
        self.formula = formula
        self._labels = base
        self.irreps_in = _factor_irreps(base, irreps)
        try:
            self.filter_ir_out = (
                None
                if filter_ir_out is None
                else tuple(Irrep.parse(value) for value in filter_ir_out)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"filter_ir_out (={filter_ir_out}) must be an iterable of Irrep values"
            ) from error
        try:
            self.filter_ir_mid = (
                None
                if filter_ir_mid is None
                else tuple(Irrep.parse(value) for value in filter_ir_mid)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"filter_ir_mid (={filter_ir_mid}) must be an iterable of Irrep values"
            ) from error
        self.irreps_out, data = _build_change_of_basis(
            formula,
            tuple(str(value) for value in self.irreps_in),
            (
                None
                if self.filter_ir_mid is None
                else tuple(str(value) for value in self.filter_ir_mid)
            ),
            (
                None
                if self.filter_ir_out is None
                else tuple(str(value) for value in self.filter_ir_out)
            ),
        )
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
