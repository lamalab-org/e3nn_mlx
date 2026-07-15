"""Batched rotations and real Wigner matrices in the upstream e3nn basis."""

from __future__ import annotations

from math import pi

from e3nn_core.irreps import Irrep, Irreps
from e3nn_core.wigner import so3_generators

from .compat import require_mlx


def _as_broadcast_arrays(*values):
    mx, _ = require_mlx()
    arrays = [value if hasattr(value, "shape") else mx.array(value, dtype=mx.float32) for value in values]
    arrays = [value if mx.issubdtype(value.dtype, mx.floating) else value.astype(mx.float32) for value in arrays]
    return mx.broadcast_arrays(*arrays)


def matrix_x(angle):
    mx, _ = require_mlx()
    (angle,) = _as_broadcast_arrays(angle)
    c, s = mx.cos(angle), mx.sin(angle)
    one, zero = mx.ones_like(angle), mx.zeros_like(angle)
    return mx.stack(
        [mx.stack([one, zero, zero], -1), mx.stack([zero, c, -s], -1), mx.stack([zero, s, c], -1)],
        axis=-2,
    )


def matrix_y(angle):
    mx, _ = require_mlx()
    (angle,) = _as_broadcast_arrays(angle)
    c, s = mx.cos(angle), mx.sin(angle)
    one, zero = mx.ones_like(angle), mx.zeros_like(angle)
    return mx.stack(
        [mx.stack([c, zero, s], -1), mx.stack([zero, one, zero], -1), mx.stack([-s, zero, c], -1)],
        axis=-2,
    )


def matrix_z(angle):
    mx, _ = require_mlx()
    (angle,) = _as_broadcast_arrays(angle)
    c, s = mx.cos(angle), mx.sin(angle)
    one, zero = mx.ones_like(angle), mx.zeros_like(angle)
    return mx.stack(
        [mx.stack([c, -s, zero], -1), mx.stack([s, c, zero], -1), mx.stack([zero, zero, one], -1)],
        axis=-2,
    )


def rotation_matrix(alpha, beta, gamma):
    """Return e3nn's active YXY Euler rotation matrix."""

    alpha, beta, gamma = _as_broadcast_arrays(alpha, beta, gamma)
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)


angles_to_matrix = rotation_matrix


def xyz_to_angles(vector):
    mx, _ = require_mlx()
    norm = mx.sqrt(mx.sum(vector * vector, axis=-1, keepdims=True))
    safe = mx.where(norm > 0, vector / mx.maximum(norm, mx.array(1e-12, dtype=vector.dtype)), mx.zeros_like(vector))
    safe = mx.clip(safe, -1.0, 1.0)
    return mx.arctan2(safe[..., 0], safe[..., 2]), mx.arccos(safe[..., 1])


def matrix_to_angles(matrix):
    mx, _ = require_mlx()
    y_axis = mx.array([0.0, 1.0, 0.0], dtype=matrix.dtype)
    direction = matrix @ y_axis
    alpha, beta = xyz_to_angles(direction)
    zero = mx.zeros_like(alpha)
    residual = mx.swapaxes(rotation_matrix(alpha, beta, zero), -1, -2) @ matrix
    gamma = mx.arctan2(residual[..., 0, 2], residual[..., 0, 0])
    return alpha, beta, gamma


def identity_angles(*shape, dtype=None):
    mx, _ = require_mlx()
    dtype = mx.float32 if dtype is None else dtype
    zero = mx.zeros(shape, dtype=dtype)
    return zero, zero, zero


def inverse_angles(alpha, beta, gamma):
    return -gamma, -beta, -alpha


def compose_angles(alpha1, beta1, gamma1, alpha2, beta2, gamma2):
    return matrix_to_angles(
        rotation_matrix(alpha1, beta1, gamma1) @ rotation_matrix(alpha2, beta2, gamma2)
    )


def axis_angle_to_matrix(axis, angle):
    """Convert normalized or non-normalized axes and angles to matrices."""

    mx, _ = require_mlx()
    angle = angle if hasattr(angle, "shape") else mx.array(angle)
    shape = mx.broadcast_shapes(tuple(axis.shape[:-1]), tuple(angle.shape))
    axis = mx.broadcast_to(axis, (*shape, 3))
    angle = mx.broadcast_to(angle, shape)
    norm = mx.sqrt(mx.sum(axis * axis, axis=-1, keepdims=True))
    default = mx.broadcast_to(mx.array([1.0, 0.0, 0.0], dtype=axis.dtype), (*shape, 3))
    axis = mx.where(norm > 0, axis / mx.maximum(norm, mx.array(1e-12, dtype=axis.dtype)), default)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s, one_c = mx.cos(angle), mx.sin(angle), 1.0 - mx.cos(angle)
    return mx.stack(
        [
            mx.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s], -1),
            mx.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s], -1),
            mx.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c], -1),
        ],
        axis=-2,
    )


def quaternion_to_matrix(quaternion):
    mx, _ = require_mlx()
    norm = mx.sqrt(mx.sum(quaternion * quaternion, axis=-1, keepdims=True))
    q = quaternion / mx.maximum(norm, mx.array(1e-12, dtype=quaternion.dtype))
    w, x, y, z = (q[..., index] for index in range(4))
    return mx.stack(
        [
            mx.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            mx.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            mx.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def axis_angle_to_quaternion(axis, angle):
    mx, _ = require_mlx()
    angle = angle if hasattr(angle, "shape") else mx.array(angle, dtype=mx.float32)
    shape = mx.broadcast_shapes(tuple(axis.shape[:-1]), tuple(angle.shape))
    axis = mx.broadcast_to(axis, (*shape, 3))
    angle = mx.broadcast_to(angle, shape)
    norm = mx.sqrt(mx.sum(axis * axis, axis=-1, keepdims=True))
    default = mx.broadcast_to(mx.array([1.0, 0.0, 0.0], dtype=axis.dtype), (*shape, 3))
    axis = mx.where(norm > 0, axis / mx.maximum(norm, mx.array(1e-12, dtype=axis.dtype)), default)
    half = 0.5 * angle
    return mx.concatenate([mx.cos(half)[..., None], axis * mx.sin(half)[..., None]], axis=-1)


def quaternion_to_axis_angle(quaternion):
    mx, _ = require_mlx()
    norm = mx.sqrt(mx.sum(quaternion * quaternion, axis=-1, keepdims=True))
    q = quaternion / mx.maximum(norm, mx.array(1e-12, dtype=quaternion.dtype))
    scalar = mx.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * mx.arccos(scalar)
    vector = q[..., 1:]
    vector_norm = mx.sqrt(mx.sum(vector * vector, axis=-1, keepdims=True))
    default = mx.broadcast_to(mx.array([1.0, 0.0, 0.0], dtype=q.dtype), vector.shape)
    axis = mx.where(vector_norm > 1e-8, vector / mx.maximum(vector_norm, 1e-12), default)
    return axis, angle


def matrix_to_axis_angle(matrix):
    mx, _ = require_mlx()
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    angle = mx.arccos(mx.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    vector = mx.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        axis=-1,
    )
    vector_norm = mx.sqrt(mx.sum(vector * vector, axis=-1, keepdims=True))
    default = mx.broadcast_to(mx.array([1.0, 0.0, 0.0], dtype=matrix.dtype), vector.shape)
    axis = mx.where(vector_norm > 1e-8, vector / mx.maximum(vector_norm, 1e-12), default)
    return axis, angle


def matrix_to_quaternion(matrix):
    return axis_angle_to_quaternion(*matrix_to_axis_angle(matrix))


def angles_to_quaternion(alpha, beta, gamma):
    alpha, beta, gamma = _as_broadcast_arrays(alpha, beta, gamma)
    return matrix_to_quaternion(rotation_matrix(alpha, beta, gamma))


def quaternion_to_angles(quaternion):
    return matrix_to_angles(quaternion_to_matrix(quaternion))


def _matrix_exp_generator(generator, angle):
    """Differentiable fixed-work matrix exponential for bounded rotation angles."""

    mx, _ = require_mlx()
    angle = mx.remainder(angle, 2.0 * pi)
    dim = generator.shape[-1]
    scaled = angle[..., None, None] * generator / 64.0
    identity = mx.eye(dim, dtype=generator.dtype)
    result = identity + scaled
    term = scaled
    for order in range(2, 19):
        term = (term @ scaled) / float(order)
        result = result + term
    for _ in range(6):
        result = result @ result
    return result


def wigner_d(l: int, alpha, beta, gamma):
    """Return the real Wigner D matrix for any non-negative integer ``l``."""

    if not isinstance(l, int) or l < 0:
        raise ValueError("l must be a non-negative integer")
    mx, _ = require_mlx()
    alpha, beta, gamma = _as_broadcast_arrays(alpha, beta, gamma)
    if l == 1:
        return rotation_matrix(alpha, beta, gamma)
    generators = mx.array(so3_generators(l), dtype=alpha.dtype)
    return (
        _matrix_exp_generator(generators[1], alpha)
        @ _matrix_exp_generator(generators[0], beta)
        @ _matrix_exp_generator(generators[1], gamma)
    )


def irrep_wigner_d(irrep: Irrep | str, alpha, beta, gamma, *, k=0):
    irrep = Irrep.parse(irrep)
    matrix = wigner_d(irrep.l, alpha, beta, gamma)
    if hasattr(k, "shape"):
        mx, _ = require_mlx()
        parity = mx.power(mx.array(float(irrep.p), dtype=matrix.dtype), k)
        return parity[..., None, None] * matrix
    return (irrep.p ** int(k)) * matrix


def _block_diag(blocks):
    mx, _ = require_mlx()
    if not blocks:
        return mx.zeros((0, 0))
    leading = mx.broadcast_shapes(*(tuple(block.shape[:-2]) for block in blocks))
    dtype = blocks[0].dtype
    total = sum(int(block.shape[-1]) for block in blocks)
    rows = []
    cursor = 0
    for block in blocks:
        width = int(block.shape[-1])
        block = mx.broadcast_to(block, (*leading, width, width))
        rows.append(
            mx.concatenate(
                [
                    mx.zeros((*leading, width, cursor), dtype=dtype),
                    block,
                    mx.zeros((*leading, width, total - cursor - width), dtype=dtype),
                ],
                axis=-1,
            )
        )
        cursor += width
    return mx.concatenate(rows, axis=-2)


def irreps_wigner_d(irreps: Irreps | str, alpha, beta, gamma, *, k=0):
    blocks = []
    for part in Irreps(irreps):
        block = irrep_wigner_d(part.ir, alpha, beta, gamma, k=k)
        blocks.extend(block for _ in range(part.mul))
    return _block_diag(blocks)


def irrep_wigner_d_from_matrix(irrep: Irrep | str, matrix):
    mx, _ = require_mlx()
    determinant = _det3x3(matrix)
    inversion = determinant < 0
    proper = mx.where(inversion[..., None, None], -matrix, matrix)
    alpha, beta, gamma = matrix_to_angles(proper)
    return irrep_wigner_d(irrep, alpha, beta, gamma, k=inversion)


def irreps_wigner_d_from_matrix(irreps: Irreps | str, matrix):
    mx, _ = require_mlx()
    determinant = _det3x3(matrix)
    inversion = determinant < 0
    proper = mx.where(inversion[..., None, None], -matrix, matrix)
    alpha, beta, gamma = matrix_to_angles(proper)
    return irreps_wigner_d(irreps, alpha, beta, gamma, k=inversion)


def _det3x3(matrix):
    return (
        matrix[..., 0, 0]
        * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1])
        - matrix[..., 0, 1]
        * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0])
        + matrix[..., 0, 2]
        * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    )
