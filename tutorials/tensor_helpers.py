"""MLX-native tensor containers used by the historical e3nn data-types tutorial.

These helpers intentionally live with the tutorial rather than in the public
``e3nn_mlx`` API.  They reproduce the small convenience layer used by the old
notebook while using :class:`e3nn_mlx.IrrepsArray` as their value container and
delegating representation theory and numerical work to
:mod:`e3nn_mlx.o3`.
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


class IrrepTensor(IrrepsArray):
    """An MLX array paired with irreducible-representation metadata.

    This is an :class:`e3nn_mlx.IrrepsArray` with the historical ``tensor`` and
    ``Rs`` names used by the tutorial. It can therefore be passed directly to
    representation-aware :mod:`e3nn_mlx.o3` operations.
    """

    def __init__(
        self,
        tensor: Any,
        representations: o3.Irreps | str | Iterable[tuple[int, ...]],
    ) -> None:
        representation_spec = representations
        legacy_rs_override = None
        if not isinstance(representations, (o3.Irreps, str)):
            representation_spec = list(representations)
            if any(len(item) == 3 and item[2] == 0 for item in representation_spec):
                legacy_rs_override = [
                    (item[0], item[1], item[2] if len(item) == 3 else 0)
                    for item in representation_spec
                ]
        super().__init__(_coerce_irreps(representation_spec), mx.array(tensor))
        object.__setattr__(self, "_legacy_rs_override", legacy_rs_override)

    @classmethod
    def from_irreps_array(cls, value: IrrepsArray) -> "IrrepTensor":
        return cls(value.array, value.irreps)

    @property
    def tensor(self):
        return self.array

    @property
    def Rs(self) -> list[tuple[int, int, int]]:
        return (
            self._legacy_rs_override
            if self._legacy_rs_override is not None
            else _legacy_rs(self.irreps)
        )

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def as_irreps_array(self) -> IrrepsArray:
        """Return this value through its public e3nn-mlx base type."""

        return self

    def __repr__(self) -> str:
        return f"IrrepTensor(shape={self.shape}, irreps={self.irreps})"


class SphericalTensor(IrrepTensor):
    """Multiplicity-free spherical coefficients with plotting conveniences."""

    def __init__(
        self,
        tensor: Any | None = None,
        lmax: int | None = None,
        p_val: int | None = None,
        p_arg: int | None = None,
    ) -> None:
        if isinstance(tensor, int) and not isinstance(tensor, bool):
            if lmax is None and (p_val is not None or p_arg is not None):
                tensor, lmax = None, tensor
            elif lmax in (-1, 1) and p_val in (-1, 1) and p_arg is None:
                # Historical fully positional form:
                # SphericalTensor(lmax, p_val, p_arg)
                tensor, lmax, p_val, p_arg = None, tensor, lmax, p_val
        if tensor is None:
            if lmax is None:
                raise TypeError("SphericalTensor requires coefficients or lmax")
            if not isinstance(lmax, int) or isinstance(lmax, bool) or lmax < 0:
                raise ValueError("lmax must be a non-negative integer")
            array = mx.zeros(((lmax + 1) ** 2,), dtype=mx.float32)
        else:
            array = mx.array(tensor)
        if array.ndim == 0:
            raise ValueError("spherical coefficients require a final coefficient dimension")
        if lmax is None:
            root = isqrt(array.shape[-1])
            if root * root != array.shape[-1]:
                raise ValueError(
                    "a complete spherical tensor must contain (lmax + 1)^2 coefficients"
                )
            lmax = root - 1
        elif not isinstance(lmax, int) or isinstance(lmax, bool) or lmax < 0:
            raise ValueError("lmax must be a non-negative integer")
        expected = (lmax + 1) ** 2
        if array.shape[-1] != expected:
            raise ValueError(
                f"lmax={lmax} requires {expected} coefficients, got {array.shape[-1]}"
            )
        if p_val is None:
            p_val = 1
        if p_arg is None:
            p_arg = -1
        if p_val not in (-1, 1) or p_arg not in (-1, 1):
            raise ValueError("p_val and p_arg must each be +1 or -1")
        object.__setattr__(self, "lmax", lmax)
        object.__setattr__(self, "p_val", p_val)
        object.__setattr__(self, "p_arg", p_arg)
        irreps = o3.Irreps(
            (1, (degree, p_val * p_arg**degree))
            for degree in range(self.lmax + 1)
        )
        super().__init__(array, irreps)

    @property
    def signal(self):
        """Historical name for the spherical-harmonic coefficient array."""

        return self.array

    def change_lmax(self, lmax: int) -> "SphericalTensor":
        """Return the same signal truncated or zero-padded to ``lmax``."""

        if not isinstance(lmax, int) or isinstance(lmax, bool) or lmax < 0:
            raise ValueError("lmax must be a non-negative integer")
        if lmax == self.lmax:
            return self
        dimension = (lmax + 1) ** 2
        if lmax < self.lmax:
            coefficients = self.array[..., :dimension]
        else:
            padding = mx.zeros(
                (*self.array.shape[:-1], dimension - self.array.shape[-1]),
                dtype=self.array.dtype,
            )
            coefficients = mx.concatenate([self.array, padding], axis=-1)
        return SphericalTensor(
            coefficients,
            lmax,
            p_val=self.p_val,
            p_arg=self.p_arg,
        )

    def __add__(self, other: object) -> "SphericalTensor":
        if not isinstance(other, SphericalTensor):
            return NotImplemented
        lmax = max(self.lmax, other.lmax)
        left = self.change_lmax(lmax)
        right = other.change_lmax(lmax)
        return SphericalTensor(left.array + right.array, lmax)

    def dot(self, other: "SphericalTensor"):
        """Coefficient-space inner product used by the historical tutorial."""

        if not isinstance(other, SphericalTensor):
            raise TypeError("dot expects another SphericalTensor")
        lmax = max(self.lmax, other.lmax)
        left = self.change_lmax(lmax)
        right = other.change_lmax(lmax)
        return mx.sum(left.array * right.array, axis=-1)

    def sum_of_diracs(self, positions: Any, values: Any):
        """Return coefficients of weighted band-limited Dirac peaks."""

        points = mx.array(positions)
        weights = mx.array(values)
        if points.ndim < 2 or points.shape[-1] != 3:
            raise ValueError("positions must have shape (..., num_points, 3)")
        if weights.ndim < 1:
            raise ValueError("values must have shape (..., num_points)")
        try:
            points, expanded_weights = mx.broadcast_arrays(
                points,
                weights[..., None],
            )
        except ValueError as error:
            raise ValueError(
                "positions and values must have broadcast-compatible leading "
                "dimensions and the same number of points"
            ) from error
        weights = expanded_weights[..., 0]
        harmonics = o3.spherical_harmonics(
            list(range(self.lmax + 1)),
            points,
            normalize=True,
            normalization="integral",
        )
        scale = 4.0 * pi / (self.lmax + 1) ** 2
        return scale * mx.sum(harmonics * weights[..., None], axis=-2)

    def with_peaks_at(self, vectors: Any, values: Any | None = None):
        """Fit coefficients whose signal has peaks at the given vectors."""

        points = mx.array(vectors)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("vectors must have shape (num_points, 3)")
        if self.p_val != 1:
            raise ValueError("with_peaks_at requires p_val=1")
        if points.shape[0] == 0:
            return mx.zeros((self.irreps.dim,), dtype=points.dtype)

        if values is None:
            targets = mx.linalg.norm(points, axis=-1)
        else:
            targets = mx.array(values)
            try:
                targets = mx.broadcast_to(targets, (points.shape[0],))
            except ValueError as error:
                raise ValueError(
                    "values must be broadcast-compatible with (num_points,)"
                ) from error

        harmonics = o3.spherical_harmonics(
            list(range(self.lmax + 1)),
            points,
            normalize=True,
            normalization="integral",
        )
        nonzero = targets != 0
        harmonics = mx.where(
            nonzero[:, None],
            harmonics,
            mx.zeros_like(harmonics),
        )
        targets = mx.where(nonzero, targets, mx.zeros_like(targets))
        gram = harmonics @ mx.swapaxes(harmonics, -1, -2)
        point_weights = mx.linalg.pinv(gram, stream=mx.cpu) @ targets
        return point_weights @ harmonics

    def D_from_angles(self, alpha: Any, beta: Any, gamma: Any, k: Any = 0):
        """Return the direct-sum Wigner-D matrix for this spherical tensor."""

        return o3.irreps_wigner_d(
            self.irreps,
            alpha,
            beta,
            gamma,
            k=k,
        )

    def __matmul__(self, other: object) -> IrrepTensor:
        """Full SO(3) tensor product, regrouped in the tutorial's degree order."""

        if not isinstance(other, SphericalTensor):
            return NotImplemented
        product = o3.FullTensorProduct(self.irreps, other.irreps)(
            self,
            other,
        )

        # Modern e3nn keeps polar and axial copies separate because their O(3)
        # parities differ. The historical tutorial used SO(3)-only metadata and
        # grouped every copy having the same degree. Within a degree it sorted
        # lower-multiplicity blocks first; the tutorial's slices rely on that
        # ordering for the l=1 cross-product block.
        by_degree: dict[int, list[tuple[Any, Any]]] = {}
        for part, chunk in zip(product.irreps, product.chunk_arrays(), strict=True):
            by_degree.setdefault(part.ir.l, []).append((part, chunk))

        representations = []
        arrays = []
        for degree in sorted(by_degree):
            entries = sorted(by_degree[degree], key=lambda entry: entry[0].mul)
            reshaped = [
                chunk.reshape(*product.leading_shape, part.mul, part.ir.dim)
                for part, chunk in entries
            ]
            grouped = (
                mx.concatenate(reshaped, axis=-2)
                if len(reshaped) > 1
                else reshaped[0]
            )
            multiplicity = sum(part.mul for part, _ in entries)
            representations.append((multiplicity, degree, 0))
            arrays.append(
                grouped.reshape(*product.leading_shape, multiplicity * (2 * degree + 1))
            )
        return IrrepTensor(mx.concatenate(arrays, axis=-1), representations)

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
        adjusted: bool = True,
        normalize: bool = True,
        normalization: str = "integral",
    ) -> "SphericalTensor":
        points = mx.array(coordinates)
        if points.ndim < 2 or points.shape[-1] != 3:
            raise ValueError("geometry must have shape (..., num_points, 3)")
        points = points.reshape(-1, 3)
        radii = mx.linalg.norm(points, axis=-1)
        nonzero = radii > 0
        if points.shape[0] == 0:
            return cls(mx.zeros(((lmax + 1) ** 2,), dtype=mx.float32), lmax)
        safe_points = mx.where(
            nonzero[:, None],
            points,
            mx.array([1.0, 0.0, 0.0], dtype=points.dtype),
        )
        coefficients = o3.spherical_harmonics(
            list(range(lmax + 1)),
            safe_points,
            normalize=normalize,
            normalization=normalization,
        )
        coefficients = mx.where(
            nonzero[:, None],
            coefficients,
            mx.zeros_like(coefficients),
        )
        if adjusted:
            # Historical e3nn's SphericalTensor.from_geometry solved for one
            # weight per point so that evaluating the resulting band-limited
            # signal at those points reproduces their radii. A plain sum has
            # the same symmetry, but generally a different magnitude.
            gram = coefficients @ mx.swapaxes(coefficients, -1, -2)
            weights = mx.linalg.pinv(gram, stream=mx.cpu) @ radii
            signal = weights @ coefficients
        else:
            signal = mx.sum(radii[:, None] * coefficients, axis=0)
        return cls(signal, lmax)

    def plot(
        self,
        *,
        relu: bool = True,
        radius: bool = True,
        res: int = 100,
        normalization: str = "integral",
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
    """Convert rank-N Cartesian tensors with ``o3.ReducedTensorProducts``.

    Unlike :class:`SphericalTensor`, this object stores a Cartesian value whose
    axes have not yet been converted to a single irrep dimension, so it is not
    itself an :class:`IrrepsArray`. Its :attr:`decomposition` is the actual
    e3nn-mlx operator and :meth:`to_irrep_tensor` returns an ``IrrepsArray``
    subclass that can be consumed directly by the rest of the library.
    """

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
        self.decomposition = o3.ReducedTensorProducts(
            self.formula,
            **{labels[0]: "1o"},
        )
        self.legacy_basis = bool(legacy_basis)
        if self.legacy_basis and self.array.ndim != 2:
            raise ValueError("legacy_basis is currently defined for rank-two Cartesian tensors")

    @property
    def change_of_basis(self):
        basis = self.decomposition.change_of_basis
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
        return self.decomposition.irreps_out

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
