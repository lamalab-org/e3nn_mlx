"""Fused Metal scatter-add and gather-transpose operations."""

from __future__ import annotations

from typing import Any

from .compat import require_mlx


_KERNEL_CACHE: dict[int, tuple[Any, Any]] = {}


def _kernels(width: int):
    mx, _ = require_mlx()
    cached = _KERNEL_CACHE.get(width)
    if cached is not None:
        return cached
    forward = mx.fast.metal_kernel(
        name=f"e3nn_scatter_sum_{width}",
        input_names=["source", "index"],
        output_names=["output"],
        source=f"""
uint tid = thread_position_in_grid.x;
uint row = tid / {width};
uint column = tid % {width};
uint destination = uint(index[row]) * {width} + column;
atomic_fetch_add_explicit(&output[destination], float(source[tid]), memory_order_relaxed);
""",
        atomic_outputs=True,
    )
    transpose = mx.fast.metal_kernel(
        name=f"e3nn_scatter_transpose_{width}",
        input_names=["cotangent", "index"],
        output_names=["gradient"],
        source=f"""
uint tid = thread_position_in_grid.x;
uint row = tid / {width};
uint column = tid % {width};
gradient[tid] = T(cotangent[uint(index[row]) * {width} + column]);
""",
    )
    _KERNEL_CACHE[width] = forward, transpose
    return forward, transpose


def make_operation(index, dim_size: int, source_shape):
    """Create a custom-differentiable scatter for a fixed index array."""

    mx, _ = require_mlx()
    width = 1
    for dimension in source_shape[1:]:
        width *= dimension
    output_shape = (dim_size, *source_shape[1:])
    element_count = source_shape[0] * width
    forward_kernel, transpose_kernel = _kernels(width)

    @mx.custom_function
    def differentiable_transpose(cotangent):
        return transpose_kernel(
            inputs=[cotangent, index],
            template=[("T", cotangent.dtype)],
            output_shapes=[source_shape],
            output_dtypes=[cotangent.dtype],
            grid=(element_count, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]

    @differentiable_transpose.vjp
    def transpose_vjp(cotangent, source_cotangent, _output):
        del cotangent
        return forward_kernel(
            inputs=[source_cotangent, index],
            template=[("T", source_cotangent.dtype)],
            output_shapes=[output_shape],
            output_dtypes=[source_cotangent.dtype],
            grid=(element_count, 1, 1),
            threadgroup=(256, 1, 1),
            init_value=0,
        )[0]

    @mx.custom_function
    def operation(source):
        return forward_kernel(
            inputs=[source, index],
            template=[("T", source.dtype)],
            output_shapes=[output_shape],
            output_dtypes=[source.dtype],
            grid=(element_count, 1, 1),
            threadgroup=(256, 1, 1),
            init_value=0,
        )[0]

    @operation.vjp
    def operation_vjp(source, cotangent, _output):
        del source
        return differentiable_transpose(cotangent)

    @operation.jvp
    def operation_jvp(source, tangent):
        del source
        output = mx.zeros(output_shape, dtype=tangent.dtype)
        return output.at[index].add(tangent)

    return operation
