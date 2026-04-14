"""Rotation and Wigner MLX helpers."""

from __future__ import annotations

from math import sqrt

from .compat import require_mlx


def rotation_matrix(alpha, beta, gamma):
    mx, _ = require_mlx()
    ca, sa = mx.cos(alpha), mx.sin(alpha)
    cb, sb = mx.cos(beta), mx.sin(beta)
    cg, sg = mx.cos(gamma), mx.sin(gamma)

    rz_alpha = mx.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry_beta = mx.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rz_gamma = mx.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
    return rz_alpha @ ry_beta @ rz_gamma


def _l2_basis_matrix():
    mx, _ = require_mlx()
    s2 = sqrt(2.0)
    return mx.array(
        [
            [0.0, 0.0, -1.0 / sqrt(6.0), 0.0, 0.0, 0.0, 0.0, 0.0, sqrt(3.0 / 2.0)],
            [1.0 / s2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / s2, 0.0, 0.0],
            [0.0, 1.0 / s2, 0.0, 1.0 / s2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / s2, 0.0, 0.0, 1.0 / s2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / s2, 0.0, 0.0, 0.0],
        ],
        dtype=mx.float32,
    )


def wigner_d(l: int, alpha, beta, gamma):
    mx, _ = require_mlx()
    rot = rotation_matrix(alpha, beta, gamma)
    if l == 0:
        return mx.ones((1, 1), dtype=rot.dtype)
    if l == 1:
        return rot
    if l == 2:
        kron = mx.kron(rot, rot)
        basis = _l2_basis_matrix().astype(rot.dtype)
        return basis @ kron @ basis.T
    raise NotImplementedError("Wigner D is implemented for l <= 2 in the MVP")
