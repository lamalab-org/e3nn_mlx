"""Smooth radial basis functions used by equivariant graph models."""

from __future__ import annotations

from math import e, pi, sqrt

from .compat import require_mlx


def soft_unit_step(x):
    r"""Smooth :math:`C^\infty` unit step ``exp(-1/x)`` for positive ``x``."""

    mx, _ = require_mlx()
    mask = x > 0
    safe = mx.where(mask, x, mx.ones_like(x))
    return mx.where(mask, mx.exp(-1.0 / safe), mx.zeros_like(x))


def soft_one_hot_linspace(
    x,
    start: float,
    end: float,
    number: int,
    basis: str,
    cutoff: bool,
):
    """Project values onto the normalized radial bases provided by upstream e3nn."""

    mx, _ = require_mlx()
    if not isinstance(cutoff, bool):
        raise ValueError("cutoff must be specified as a boolean")
    if number <= 0:
        raise ValueError("number must be positive")
    if not end > start:
        raise ValueError("end must be greater than start")
    if not cutoff and number < 2 and basis in ("gaussian", "cosine", "smooth_finite"):
        raise ValueError("number must be at least two without cutoff for this basis")

    if basis == "fourier":
        normalized = (x[..., None] - start) / (end - start)
        if cutoff:
            frequencies = mx.arange(1, number + 1, dtype=x.dtype)
            inside = (normalized > 0) & (normalized < 1)
            return (
                mx.sin(pi * frequencies * normalized)
                / sqrt(0.25 + number / 2)
                * inside
            )
        frequencies = mx.arange(number, dtype=x.dtype)
        return mx.cos(pi * frequencies * normalized) / sqrt(0.25 + number / 2)

    if basis == "bessel":
        shifted = x[..., None] - start
        width = end - start
        roots = mx.arange(1, number + 1, dtype=x.dtype) * pi
        safe = mx.where(shifted != 0, shifted, mx.ones_like(shifted))
        result = sqrt(2.0 / width) * mx.sin(roots * shifted / width) / safe
        limit = sqrt(2.0 / width) * roots / width
        result = mx.where(shifted != 0, result, limit)
        if cutoff:
            result = result * ((shifted > 0) & (shifted / width < 1))
        return result

    count = number + 2 if cutoff else number
    values = mx.linspace(start, end, count, dtype=x.dtype)
    step = values[1] - values[0]
    if cutoff:
        values = values[1:-1]
    diff = (x[..., None] - values) / step

    if basis == "gaussian":
        return mx.exp(-(diff**2)) / 1.12
    if basis == "cosine":
        return mx.cos(pi / 2 * diff) * ((diff < 1) & (diff > -1))
    if basis == "smooth_finite":
        return 1.14136 * e**2 * soft_unit_step(diff + 1) * soft_unit_step(1 - diff)
    raise ValueError(f'basis="{basis}" is not a valid entry')


def smooth_cutoff(x):
    """Cosine cutoff equal to one below 1/2 and zero above one."""

    mx, _ = require_mlx()
    u = 2 * (x - 1)
    middle = (1 - mx.cos(pi * u)) / 2
    return mx.where(u > 0, mx.zeros_like(x), mx.where(u < -1, mx.ones_like(x), middle))
