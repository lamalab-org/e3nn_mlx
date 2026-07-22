"""Generated Metal kernels for batches of real spherical harmonics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import sqrt
from typing import Any

from e3nn_core.cg import wigner_3j

from .compat import require_mlx


_KERNEL_CACHE: dict[tuple[tuple[int, ...], bool, tuple[float, ...]], tuple[Any, Any]] = {}


def _literal(value: float) -> str:
    literal = f"{value:.17g}"
    if "." not in literal and "e" not in literal:
        literal += ".0"
    return literal + "f"


def _program(
    degrees: tuple[int, ...], normalize: bool, scales: tuple[float, ...], *, backward: bool
) -> str:
    lmax = max(degrees)
    lines = [
        "uint item = thread_position_in_grid.x;",
        "uint input_base = item * 3;",
        "float x0 = float(vectors[input_base]);",
        "float x1 = float(vectors[input_base + 1]);",
        "float x2 = float(vectors[input_base + 2]);",
    ]
    if normalize:
        lines += [
            "float radius = sqrt(x0*x0 + x1*x1 + x2*x2);",
            "float inverse_radius = radius > 0.0f ? 1.0f / max(radius, 1.0e-12f) : 0.0f;",
            "float n0 = x0 * inverse_radius;",
            "float n1 = x1 * inverse_radius;",
            "float n2 = x2 * inverse_radius;",
        ]
    else:
        lines += ["float n0 = x0;", "float n1 = x1;", "float n2 = x2;"]
    lines.append("float y0_0 = 1.0f;")
    if backward:
        lines += [f"float dy0_0_{k} = 0.0f;" for k in range(3)]
    if lmax:
        root3 = sqrt(3.0)
        for component in range(3):
            lines.append(f"float y1_{component} = {_literal(root3)} * n{component};")
            if backward:
                for k in range(3):
                    if normalize:
                        derivative = (
                            f"inverse_radius * ({'1.0f' if component == k else '0.0f'} "
                            f"- n{component} * n{k})"
                        )
                    else:
                        derivative = "1.0f" if component == k else "0.0f"
                    lines.append(
                        f"float dy1_{component}_{k} = {_literal(root3)} * ({derivative});"
                    )

    for l in range(1, lmax):
        coefficients = wigner_3j(l, 1, l + 1)
        factor = (2 * l + 3) / sqrt(3.0 * (l + 1))
        for c in range(2 * (l + 1) + 1):
            terms: list[tuple[int, int, float]] = []
            for a in range(2 * l + 1):
                for b in range(3):
                    coefficient = float(coefficients[a][b][c]) * factor
                    if coefficient != 0.0:
                        terms.append((a, b, coefficient))
            expression = " + ".join(
                f"{_literal(coefficient)} * y{l}_{a} * y1_{b}"
                for a, b, coefficient in terms
            ) or "0.0f"
            lines.append(f"float y{l + 1}_{c} = {expression};")
            if backward:
                for k in range(3):
                    derivative = " + ".join(
                        f"{_literal(coefficient)} * (dy{l}_{a}_{k} * y1_{b} + y{l}_{a} * dy1_{b}_{k})"
                        for a, b, coefficient in terms
                    ) or "0.0f"
                    lines.append(f"float dy{l + 1}_{c}_{k} = {derivative};")

    output_offset = 0
    if backward:
        for k in range(3):
            lines.append(f"float gradient{k} = 0.0f;")
        for degree, scale in zip(degrees, scales, strict=True):
            for component in range(2 * degree + 1):
                for k in range(3):
                    lines.append(
                        f"gradient{k} += float(cotangent[item * {sum(2*d+1 for d in degrees)} + {output_offset}]) "
                        f"* {_literal(scale)} * dy{degree}_{component}_{k};"
                    )
                output_offset += 1
        lines += [f"vector_gradient[input_base + {k}] = T(gradient{k});" for k in range(3)]
    else:
        output_dim = sum(2 * degree + 1 for degree in degrees)
        for degree, scale in zip(degrees, scales, strict=True):
            for component in range(2 * degree + 1):
                lines.append(
                    f"output[item * {output_dim} + {output_offset}] = "
                    f"T({_literal(scale)} * y{degree}_{component});"
                )
                output_offset += 1
    return "\n".join(lines)


def _kernels(degrees: tuple[int, ...], normalize: bool, scales: tuple[float, ...]):
    mx, _ = require_mlx()
    key = (degrees, normalize, scales)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    suffix = "_".join(map(str, degrees))
    forward = mx.fast.metal_kernel(
        name=f"e3nn_sh_forward_{suffix}_{int(normalize)}",
        input_names=["vectors"],
        output_names=["output"],
        source=_program(degrees, normalize, scales, backward=False),
    )
    backward = mx.fast.metal_kernel(
        name=f"e3nn_sh_backward_{suffix}_{int(normalize)}",
        input_names=["vectors", "cotangent"],
        output_names=["vector_gradient"],
        source=_program(degrees, normalize, scales, backward=True),
    )
    _KERNEL_CACHE[key] = forward, backward
    return forward, backward


def make_operation(
    degrees: Sequence[int],
    normalize: bool,
    scales: Sequence[float],
    general: Callable[[Any], Any],
):
    """Create a fused per-vector spherical-harmonics operation."""

    mx, _ = require_mlx()
    degrees = tuple(degrees)
    scales = tuple(scales)
    output_dim = sum(2 * degree + 1 for degree in degrees)
    forward_kernel, backward_kernel = _kernels(degrees, normalize, scales)

    @mx.custom_function
    def differentiable_backward(vectors, cotangent):
        return backward_kernel(
            inputs=[vectors, cotangent],
            template=[("T", vectors.dtype)],
            output_shapes=[vectors.shape],
            output_dtypes=[vectors.dtype],
            grid=(vectors.shape[0], 1, 1),
            threadgroup=(256, 1, 1),
        )[0]

    @differentiable_backward.vjp
    def differentiable_backward_vjp(primals, cotangent, _output):
        vectors, output_cotangent = primals

        def general_backward(x, dy):
            _, (gradient,) = mx.vjp(general, (x,), (dy,))
            return gradient

        _, gradients = mx.vjp(general_backward, (vectors, output_cotangent), (cotangent,))
        return gradients

    @mx.custom_function
    def operation(vectors):
        return forward_kernel(
            inputs=[vectors],
            template=[("T", vectors.dtype)],
            output_shapes=[(vectors.shape[0], output_dim)],
            output_dtypes=[vectors.dtype],
            grid=(vectors.shape[0], 1, 1),
            threadgroup=(256, 1, 1),
        )[0]

    @operation.vjp
    def operation_vjp(primals, cotangent, _output):
        return differentiable_backward(primals, cotangent)

    @operation.jvp
    def operation_jvp(primals, tangents):
        _, tangent = mx.jvp(general, (primals,), (tangents,))
        return tangent

    return operation
