"""Wigner bookkeeping metadata."""

from __future__ import annotations

import functools
from math import sqrt


@functools.lru_cache(maxsize=None)
def change_basis_real_to_complex(l: int) -> tuple[tuple[complex, ...], ...]:
    if not isinstance(l, int) or l < 0:
        raise ValueError("l must be a non-negative integer")
    dim = 2 * l + 1
    q = [[0j for _ in range(dim)] for _ in range(dim)]
    s = 2.0**-0.5
    for m in range(-l, 0):
        q[l + m][l + abs(m)] = s
        q[l + m][l - abs(m)] = -1j * s
    q[l][l] = 1.0 + 0j
    for m in range(1, l + 1):
        q[l + m][l + abs(m)] = ((-1) ** m) * s
        q[l + m][l - abs(m)] = 1j * ((-1) ** m) * s
    phase = (-1j) ** l
    return tuple(tuple(phase * value for value in row) for row in q)


@functools.lru_cache(maxsize=None)
def su2_generators(l: int | float) -> tuple[tuple[tuple[complex, ...], ...], ...]:
    """Return the three anti-Hermitian SU(2) generators used by e3nn."""

    if not isinstance(l, (int, float)) or l < 0 or not float(2 * l).is_integer():
        raise ValueError("l must be a non-negative integer or half-integer")
    l = float(l)
    dim = int(2 * l + 1)
    raising = [[0j for _ in range(dim)] for _ in range(dim)]
    lowering = [[0j for _ in range(dim)] for _ in range(dim)]
    magnetic = [-l + index for index in range(dim)]
    for index, m in enumerate(magnetic[:-1]):
        raising[index + 1][index] = -sqrt(l * (l + 1) - m * (m + 1))
    for index, m in enumerate(magnetic[1:]):
        lowering[index][index + 1] = sqrt(l * (l + 1) - m * (m - 1))

    generators = [[[0j for _ in range(dim)] for _ in range(dim)] for _ in range(3)]
    for row in range(dim):
        for column in range(dim):
            generators[0][row][column] = 0.5 * (raising[row][column] + lowering[row][column])
            generators[2][row][column] = -0.5j * (raising[row][column] - lowering[row][column])
    for index, m in enumerate(magnetic):
        generators[1][index][index] = 1j * m
    return tuple(tuple(tuple(row) for row in generator) for generator in generators)


@functools.lru_cache(maxsize=None)
def so3_generators(l: int) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Return e3nn's real anti-symmetric SO(3) generators."""

    q = change_basis_real_to_complex(l)
    x = su2_generators(l)
    dim = 2 * l + 1
    # conj(q.T), so that adjoint[row][left] == q[left][row].conjugate()
    adjoint = [[q[left][row].conjugate() for left in range(dim)] for row in range(dim)]
    out = [[[0.0 for _ in range(dim)] for _ in range(dim)] for _ in range(3)]
    for axis in range(3):
        axis_generator = x[axis]
        # Evaluate conj(q.T) @ x @ q as two matrix products rather than one
        # quadruple loop: O(dim ** 3) instead of O(dim ** 4).
        partial = [
            [sum(row[left] * axis_generator[left][column] for left in range(dim)) for column in range(dim)]
            for row in adjoint
        ]
        for row in range(dim):
            partial_row = partial[row]
            for column in range(dim):
                value = sum(partial_row[right] * q[right][column] for right in range(dim))
                if abs(value.imag) > 1e-10:
                    raise ArithmeticError("real-basis generator has a non-negligible imaginary component")
                out[axis][row][column] = value.real
    return tuple(tuple(tuple(row) for row in generator) for generator in out)
