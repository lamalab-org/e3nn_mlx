"""MLX-native tensor containers used by the historical e3nn data-types tutorial.

These helpers intentionally live with the tutorial rather than in the public
``e3nn_mlx`` API.  They reproduce the small convenience layer used by the old
notebook while delegating representation theory and numerical work to the
modern :mod:`e3nn_mlx.o3` implementation.
"""

from __future__ import annotations

from math import isqrt, pi, sqrt
from typing import Any, Iterable

import mlx.core as mx

from e3nn_mlx import IrrepsArray, o3


_TENSOR_LABELS = "ijklmnopqrstuvwxyzabcdefgh"


def _legacy_rs(irreps: o3.Irreps) -> list[tuple[int, int, int]]:
    return [(part.mul, part.ir.l, part.ir.p) for part in irreps]


def _coerce_irreps(spec: o3.Irreps | str | Iterable[tuple[int, ...]]) -> o3.Irreps:
    if isinstance(spec, (o3.Irreps, str)):
        return o3.Irreps(spec)
    parts = []
    for item in spec:
        if len(item) == 2:
            multiplicity, degree = item
            parity = (-1) ** degree
        elif len(item) == 3:
            multiplicity, degree, parity = item
            # Historical representation lists used p=0 to mean SO(3)-only.
            # Natural spherical parity gives that metadata an O(3) embedding
            # without changing any proper-rotation behavior.
            parity = (-1) ** degree if parity == 0 else parity
        else:
            raise ValueError("representation entries must be (mul, l) or (mul, l, p)")
        parts.append((multiplicity, (degree, parity)))
    return o3.Irreps(parts)


class IrrepTensor:
    """An MLX array paired with irreducible-representation metadata.

    ``Rs`` is provided for compatibility with the old tutorial. New code can
    use :attr:`irreps` and :meth:`as_irreps_array` directly.
    """

    def __init__(
        self,
        tensor: Any,
        representations: o3.Irreps | str | Iterable[tuple[int, ...]],
    ) -> None:
        self.irreps = _coerce_irreps(representations)
        self.array = mx.array(tensor)
        if self.array.ndim == 0 or self.array.shape[-1] != self.irreps.dim:
            raise ValueError(
                f"last tensor dimension {self.array.shape[-1] if self.array.ndim else None} "
                f"does not match irreps dimension {self.irreps.dim}"
            )

    @classmethod
    def from_irreps_array(cls, value: IrrepsArray) -> "IrrepTensor":
        return cls(value.array, value.irreps)

    @property
    def tensor(self):
        return self.array

    @property
    def Rs(self) -> list[tuple[int, int, int]]:
        return _legacy_rs(self.irreps)

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def as_irreps_array(self) -> IrrepsArray:
        return IrrepsArray(self.irreps, self.array)

    def __repr__(self) -> str:
        return f"IrrepTensor(shape={self.shape}, irreps={self.irreps})"


class SphericalTensor(IrrepTensor):
    """Multiplicity-free spherical coefficients with plotting conveniences."""

    def __init__(self, tensor: Any) -> None:
        array = mx.array(tensor)
        if array.ndim == 0:
            raise ValueError("spherical coefficients require a final coefficient dimension")
        root = isqrt(array.shape[-1])
        if root * root != array.shape[-1]:
            raise ValueError("a complete spherical tensor must contain (lmax + 1)^2 coefficients")
        self.lmax = root - 1
        super().__init__(array, o3.Irreps.spherical_harmonics(self.lmax))

    @classmethod
    def from_irrep_tensor(
        cls,
        value: IrrepTensor | IrrepsArray,
    ) -> "SphericalTensor":
        typed = value.as_irreps_array() if isinstance(value, IrrepTensor) else value
        if not isinstance(typed, IrrepsArray):
            raise TypeError("from_irrep_tensor expects IrrepTensor or IrrepsArray")

        by_degree = {}
        for part, chunk in zip(typed.irreps, typed.chunk_arrays(), strict=True):
            if part.mul != 1 or part.ir.l in by_degree:
                raise ValueError("a spherical tensor can contain at most one copy of each degree")
            by_degree[part.ir.l] = chunk
        lmax = max(by_degree, default=0)
        coefficients = [
            by_degree.get(
                degree,
                mx.zeros((*typed.leading_shape, 2 * degree + 1), dtype=typed.dtype),
            )
            for degree in range(lmax + 1)
        ]
        return cls(mx.concatenate(coefficients, axis=-1))

    @classmethod
    def from_geometry(
        cls,
        coordinates: Any,
        lmax: int,
        *,
        normalize: bool = True,
        normalization: str = "component",
    ) -> "SphericalTensor":
        points = mx.array(coordinates)
        if points.ndim < 2 or points.shape[-1] != 3:
            raise ValueError("geometry must have shape (..., num_points, 3)")
        coefficients = o3.spherical_harmonics(
            list(range(lmax + 1)),
            points,
            normalize=normalize,
            normalization=normalization,
        )
        return cls(mx.sum(coefficients, axis=-2))

    def plot(
        self,
        *,
        relu: bool = True,
        radius: bool = True,
        res: int = 100,
        normalization: str = "component",
        legacy_axes: bool = False,
    ):
        """Return surface coordinates and values suitable for Plotly.

        The helper deliberately returns MLX arrays. Convert them with
        ``numpy.asarray`` before passing them to plotting libraries.

        Set ``legacy_axes=True`` to reproduce the historical tutorial's
        ``(y, z, x)`` degree-one display convention. The default follows the
        current e3nn-mlx ``(x, y, z)`` basis.
        """

        if self.array.ndim != 1:
            raise ValueError("plot expects one unbatched spherical tensor")
        if res <= 0:
            raise ValueError("res must be positive")
        resolution = max(2 * (self.lmax + 1), res, 2)
        betas = mx.linspace(0.0, pi, resolution)
        # Include both 0 and 2*pi. They represent the same meridian, but Plotly
        # needs the duplicate column to close a Surface mesh.
        alphas = mx.linspace(0.0, 2.0 * pi, resolution + 1)
        harmonics = o3.spherical_harmonics_alpha_beta(
            list(range(self.lmax + 1)),
            alphas[None, :],
            betas[:, None],
            normalization="integral",
        )
        if normalization == "component":
            degree_scales = [
                sqrt(4.0 * pi / (2 * degree + 1)) / sqrt(self.lmax + 1)
                for degree in range(self.lmax + 1)
            ]
        elif normalization == "norm":
            degree_scales = [
                sqrt(4.0 * pi) / sqrt(self.lmax + 1)
                for _ in range(self.lmax + 1)
            ]
        elif normalization == "integral":
            degree_scales = [1.0 for _ in range(self.lmax + 1)]
        else:
            raise ValueError("normalization must be 'component', 'norm', or 'integral'")
        scales = mx.array(
            [
                scale
                for degree, scale in enumerate(degree_scales)
                for _ in range(2 * degree + 1)
            ],
            dtype=self.array.dtype,
        )
        values = harmonics @ (self.array * scales)
        grid = o3.angles_to_xyz(alphas[None, :], betas[:, None])
        if legacy_axes:
            # Map physical x -> displayed y, y -> displayed z, and z ->
            # displayed x, matching the old tutorial's (y, z, x) labels.
            grid = grid[..., [2, 0, 1]]
        colors = mx.maximum(values, 0.0) if relu else values
        if radius:
            radial = colors if relu else mx.abs(values)
            surface = grid * radial[..., None]
        else:
            surface = grid
        return surface, colors


class CartesianTensor:
    """Convert rank-N Cartesian tensors into the coupled irrep basis."""

    def __init__(
        self,
        tensor: Any,
        formula: str | None = None,
        *,
        legacy_basis: bool = False,
    ) -> None:
        self.array = mx.array(tensor)
        if self.array.ndim < 1 or any(size != 3 for size in self.array.shape):
            raise ValueError("CartesianTensor expects an unbatched tensor with every axis of size 3")
        if self.array.ndim > len(_TENSOR_LABELS):
            raise ValueError("Cartesian tensor rank is too large")
        labels = _TENSOR_LABELS[: self.array.ndim]
        self.formula = labels if formula is None else formula.replace(" ", "")
        if self.formula.split("=", 1)[0].lstrip("+-") != labels:
            raise ValueError(f"formula must start with {labels!r} for a rank-{self.array.ndim} tensor")
        self._decomposition = o3.ReducedTensorProducts(self.formula, **{labels[0]: "1o"})
        self.legacy_basis = bool(legacy_basis)
        if self.legacy_basis and self.array.ndim != 2:
            raise ValueError("legacy_basis is currently defined for rank-two Cartesian tensors")

    @property
    def change_of_basis(self):
        basis = self._decomposition.change_of_basis
        if not self.legacy_basis:
            return basis

        # The historical tutorial first mapped Cartesian (x, y, z) components
        # to its irreducible (y, z, x) vector basis. Expressed as a basis acting
        # directly on the original Cartesian tensor, this permutes both input
        # axes by the inverse cycle (z, x, y).
        inverse_cycle = mx.array([2, 0, 1])
        basis = basis[:, inverse_cycle][:, :, inverse_cycle]

        # The old numerical Wigner-3j construction chose the opposite phase for
        # the rank-two l=2 coupling. Irrep phases are conventional, but applying
        # it here reproduces the tutorial's printed transformation exactly.
        phases = mx.concatenate(
            [
                mx.full((part.mul * part.ir.dim,), -1.0 if part.ir.l == 2 else 1.0)
                for part in self.irreps
            ]
        ).astype(basis.dtype)
        return phases[:, None, None] * basis

    @property
    def irreps(self) -> o3.Irreps:
        return self._decomposition.irreps_out

    @property
    def Rs(self) -> list[tuple[int, int, int]]:
        return _legacy_rs(self.irreps)

    def to_irrep_transformation(self):
        return self.Rs, self.change_of_basis

    def to_irrep_tensor(self) -> IrrepTensor:
        basis = self.change_of_basis.reshape(self.irreps.dim, -1)
        coefficients = basis @ self.array.reshape(-1)
        return IrrepTensor(coefficients, self.irreps)

    def project_to_cartesian(self):
        """Project onto the requested symmetry class in Cartesian coordinates."""

        basis = self.change_of_basis.reshape(self.irreps.dim, -1)
        coefficients = basis @ self.array.reshape(-1)
        return (mx.swapaxes(basis, -1, -2) @ coefficients).reshape(self.array.shape)

    def __repr__(self) -> str:
        return (
            f"CartesianTensor(shape={self.array.shape}, formula={self.formula!r}, "
            f"irreps={self.irreps}, legacy_basis={self.legacy_basis})"
        )
