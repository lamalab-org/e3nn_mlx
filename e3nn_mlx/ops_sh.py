"""Real spherical harmonics generated from the canonical e3nn Wigner basis."""

from __future__ import annotations

from math import pi, sqrt
from typing import Sequence

from e3nn_core.cg import wigner_3j
from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from .compat import require_mlx
from .irreps_array import IrrepsArray


def _parse_degrees(spec, input_parity: int) -> tuple[list[int], Irreps]:
    if isinstance(spec, int):
        degrees = [spec]
    elif isinstance(spec, (str, Irreps)):
        requested = Irreps(spec)
        degrees = []
        for part in requested:
            expected_parity = input_parity ** part.ir.l
            if part.ir.p != expected_parity:
                raise ValueError(
                    f"spherical harmonic l={part.ir.l} requires parity {expected_parity}, got {part.ir.p}"
                )
            degrees.extend([part.ir.l] * part.mul)
    else:
        degrees = [int(l) for l in spec]
    if not degrees:
        return [], Irreps()
    if any(l < 0 for l in degrees):
        raise ValueError("spherical harmonic degrees must be non-negative")
    return degrees, Irreps(MulIrrep(1, Irrep(l, input_parity**l)) for l in degrees).simplify()


def _normalized_vectors(vectors, normalize: bool):
    if not normalize:
        return vectors
    mx, _ = require_mlx()
    norm = mx.sqrt(mx.sum(vectors * vectors, axis=-1, keepdims=True))
    safe_norm = mx.maximum(norm, mx.array(1e-12, dtype=vectors.dtype))
    return mx.where(norm > 0, vectors / safe_norm, mx.zeros_like(vectors))


def _component_harmonics(lmax: int, vectors):
    mx, _ = require_mlx()
    outputs = [mx.ones((*vectors.shape[:-1], 1), dtype=vectors.dtype)]
    if lmax == 0:
        return outputs
    y1 = sqrt(3.0) * vectors
    outputs.append(y1)
    previous = y1
    for l in range(1, lmax):
        coefficients = mx.array(wigner_3j(l, 1, l + 1), dtype=vectors.dtype)
        scale = (2 * l + 3) / sqrt(3.0 * (l + 1))
        previous = scale * mx.einsum("...a,...b,abc->...c", previous, y1, coefficients)
        outputs.append(previous.astype(vectors.dtype))
    return outputs


def _scale_for_normalization(l: int, normalization: str) -> float:
    if normalization == "component":
        return 1.0
    if normalization == "norm":
        return 1.0 / sqrt(2 * l + 1)
    if normalization == "integral":
        return 1.0 / sqrt(4.0 * pi)
    raise ValueError("normalization must be 'component', 'norm', or 'integral'")


def spherical_harmonics(
    degrees: int | Sequence[int] | Irreps | str,
    vectors,
    *,
    normalize: bool = True,
    normalization: str = "component",
):
    """Evaluate real spherical harmonics in the canonical e3nn basis.

    A raw MLX input returns a raw MLX array.  An :class:`IrrepsArray` input
    returns an :class:`IrrepsArray` and uses its vector parity.
    """

    mx, _ = require_mlx()
    wrapped = isinstance(vectors, IrrepsArray)
    if wrapped:
        if len(vectors.irreps) != 1 or vectors.irreps[0].mul != 1 or vectors.irreps[0].ir.l != 1:
            raise ValueError("spherical_harmonics expects a single vector irrep (1o or 1e)")
        input_parity = vectors.irreps[0].ir.p
        raw_vectors = vectors.array
    else:
        input_parity = -1
        raw_vectors = vectors
    if raw_vectors.ndim < 1 or raw_vectors.shape[-1] != 3:
        raise ValueError("spherical_harmonics expects input shape (..., 3)")

    parsed_degrees, irreps_out = _parse_degrees(degrees, input_parity)
    if not parsed_degrees:
        result = mx.zeros((*raw_vectors.shape[:-1], 0), dtype=raw_vectors.dtype)
        return IrrepsArray(irreps_out, result) if wrapped else result

    normalized_vectors = _normalized_vectors(raw_vectors, normalize)
    all_harmonics = _component_harmonics(max(parsed_degrees), normalized_vectors)
    selected = [all_harmonics[l] * _scale_for_normalization(l, normalization) for l in parsed_degrees]
    result = mx.concatenate(selected, axis=-1).astype(raw_vectors.dtype)
    return IrrepsArray(irreps_out, result) if wrapped else result


def sh(degrees, vectors, *, normalize: bool = True, normalization: str = "component"):
    return spherical_harmonics(degrees, vectors, normalize=normalize, normalization=normalization)
