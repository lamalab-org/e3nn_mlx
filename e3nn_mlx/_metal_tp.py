"""Generated sparse Metal kernels for static tensor-product instruction plans."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from e3nn_core.instructions import TensorProductInstruction
from e3nn_core.irreps import Irreps

from .compat import require_mlx


@dataclass(frozen=True, slots=True)
class TensorProductKernelMetadata:
    """Padded sparse paths consumed by one generated tensor-product kernel."""

    left_indices: Any
    right_indices: Any
    weight_indices: Any
    coefficients: Any
    width: int
    output_dim: int
    weighted: bool


@dataclass(frozen=True, slots=True)
class ChannelTensorProductMetadata:
    left_offsets: Any
    left_strides: Any
    right_indices: Any
    weight_offsets: Any
    weight_strides: Any
    coefficients: Any
    output_offsets: Any
    output_strides: Any
    width: int
    rows: int
    channels: int
    output_dim: int


_KERNEL_CACHE: dict[tuple[int, int, int, int, bool], tuple[Any, Any]] = {}
_CHANNEL_KERNEL_CACHE: dict[
    tuple[int, int, int, int, int, int], tuple[Any, Any]
] = {}


def _block_offsets(irreps: Irreps) -> tuple[int, ...]:
    offsets = []
    cursor = 0
    for part in irreps:
        offsets.append(cursor)
        cursor += part.dim
    return tuple(offsets)


def _default_output_maps(
    instructions: Sequence[TensorProductInstruction], irreps_out: Irreps
) -> tuple[tuple[int, ...], ...]:
    offsets = _block_offsets(irreps_out)
    maps = []
    for instruction in instructions:
        base = offsets[instruction.output_index]
        maps.append(
            tuple(
                base + index
                for index in range(
                    irreps_out[instruction.output_index].dim
                )
            )
        )
    return tuple(maps)


def _connection_paths(
    instruction: TensorProductInstruction,
) -> tuple[tuple[int, int, int, int | None], ...]:
    """Return ``(u, v, output_mul, local_weight)`` connection paths."""

    mode = instruction.mode
    shape = instruction.path_shape
    if mode == "uvw":
        left_mul, right_mul, output_mul = shape
        return tuple(
            (u, v, w, (u * right_mul + v) * output_mul + w)
            for w in range(output_mul)
            for u in range(left_mul)
            for v in range(right_mul)
        )
    if mode == "uvu":
        left_mul, right_mul = shape
        return tuple(
            (u, v, u, u * right_mul + v)
            for u in range(left_mul)
            for v in range(right_mul)
        )
    if mode == "uvv":
        left_mul, right_mul = shape
        return tuple(
            (u, v, v, u * right_mul + v)
            for v in range(right_mul)
            for u in range(left_mul)
        )
    if mode == "uuw":
        diagonal_mul, output_mul = shape
        return tuple(
            (u, u, w, u * output_mul + w)
            for w in range(output_mul)
            for u in range(diagonal_mul)
        )
    if mode == "uuu":
        (diagonal_mul,) = shape
        return tuple((u, u, u, u) for u in range(diagonal_mul))
    if mode == "uvuv":
        left_mul, right_mul = shape
        return tuple(
            (u, v, u * right_mul + v, u * right_mul + v)
            for u in range(left_mul)
            for v in range(right_mul)
        )
    if mode in ("uvu<v", "u<vw"):
        # Both upper-triangle modes encode q = mul * (mul - 1) / 2.
        pair_count = shape[0]
        input_mul = int((1 + (1 + 8 * pair_count) ** 0.5) / 2)
        upper = [
            (u, v)
            for u in range(input_mul)
            for v in range(u + 1, input_mul)
        ]
        if mode == "uvu<v":
            return tuple((u, v, q, q) for q, (u, v) in enumerate(upper))
        pair_count, output_mul = shape
        if pair_count != len(upper):
            raise ValueError("invalid u<vw upper-triangle path shape")
        return tuple(
            (u, v, w, q * output_mul + w)
            for w in range(output_mul)
            for q, (u, v) in enumerate(upper)
        )
    raise ValueError(f"unsupported Metal tensor-product mode {mode!r}")


def build_metadata(
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    irreps_out: Irreps,
    instructions: Sequence[TensorProductInstruction],
    weight_slices: dict[int, slice],
    *,
    output_maps: Sequence[Sequence[int]] | None = None,
) -> TensorProductKernelMetadata:
    """Expand normalized instructions into sparse scalar contraction paths."""

    mx, _ = require_mlx()
    left_offsets = _block_offsets(irreps_in1)
    right_offsets = _block_offsets(irreps_in2)
    if output_maps is None:
        output_maps = _default_output_maps(instructions, irreps_out)
    if len(output_maps) != len(instructions):
        raise ValueError("expected one Metal output map per instruction")

    output_dim = irreps_out.dim
    rows: list[list[tuple[int, int, int, float]]] = [
        [] for _ in range(output_dim)
    ]
    weighted = bool(weight_slices)
    for instruction_index, (instruction, output_map) in enumerate(
        zip(instructions, output_maps, strict=True)
    ):
        left_part = irreps_in1[instruction.input1_index]
        right_part = irreps_in2[instruction.input2_index]
        connection_paths = _connection_paths(instruction)
        if not connection_paths:
            continue
        expected_output_size = (
            max(path[2] for path in connection_paths) + 1
        ) * instruction.ir_out.dim
        if len(output_map) != expected_output_size:
            raise ValueError(
                "Metal output map does not match instruction output shape"
            )
        coefficient_scale = instruction.normalization.scale()
        cg = instruction.ir_in1, instruction.ir_in2, instruction.ir_out
        from .ops_tp import _cg_array_data

        coefficients = _cg_array_data(*cg)
        weight_slice = weight_slices.get(instruction_index)
        for u, v, output_mul, local_weight in connection_paths:
            weight_index = -1
            if instruction.has_weight:
                if weight_slice is None or local_weight is None:
                    raise ValueError("missing weight path for weighted instruction")
                weight_index = weight_slice.start + local_weight
            for a in range(instruction.ir_in1.dim):
                left_index = (
                    left_offsets[instruction.input1_index]
                    + u * instruction.ir_in1.dim
                    + a
                )
                for b in range(instruction.ir_in2.dim):
                    right_index = (
                        right_offsets[instruction.input2_index]
                        + v * instruction.ir_in2.dim
                        + b
                    )
                    for c, coefficient in enumerate(coefficients[a][b]):
                        value = float(coefficient) * coefficient_scale
                        if value == 0.0:
                            continue
                        output_index = output_map[
                            output_mul * instruction.ir_out.dim + c
                        ]
                        rows[output_index].append(
                            (left_index, right_index, weight_index, value)
                        )

    width = max((len(row) for row in rows), default=0)
    left_data = [[0] * width for _ in range(output_dim)]
    right_data = [[0] * width for _ in range(output_dim)]
    weight_data = [[0] * width for _ in range(output_dim)]
    coefficient_data = [[0.0] * width for _ in range(output_dim)]
    for output_index, row in enumerate(rows):
        for column, (left, right, weight, coefficient) in enumerate(row):
            left_data[output_index][column] = left
            right_data[output_index][column] = right
            weight_data[output_index][column] = max(weight, 0)
            coefficient_data[output_index][column] = coefficient
    return TensorProductKernelMetadata(
        left_indices=mx.array(left_data, dtype=mx.int32),
        right_indices=mx.array(right_data, dtype=mx.int32),
        weight_indices=mx.array(weight_data, dtype=mx.int32),
        coefficients=mx.array(coefficient_data, dtype=mx.float32),
        width=width,
        output_dim=output_dim,
        weighted=weighted,
    )


def build_channel_metadata(
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    irreps_out: Irreps,
    instructions: Sequence[TensorProductInstruction],
    weight_slices: dict[int, slice],
) -> ChannelTensorProductMetadata | None:
    """Build channel-local metadata for weighted ``uvu`` instruction groups."""

    mx, _ = require_mlx()
    from .ops_tp import _cg_array_data

    if not instructions or any(
        instruction.mode != "uvu" or not instruction.has_weight
        for instruction in instructions
    ):
        return None
    output_indices = [instruction.output_index for instruction in instructions]
    if len(set(output_indices)) != len(output_indices):
        return None
    channels = irreps_in1[instructions[0].input1_index].mul
    if any(
        irreps_in1[instruction.input1_index].mul != channels
        or irreps_out[instruction.output_index].mul != channels
        for instruction in instructions
    ):
        return None
    left_blocks = _block_offsets(irreps_in1)
    right_blocks = _block_offsets(irreps_in2)
    output_blocks = _block_offsets(irreps_out)
    rows: list[
        tuple[int, int, list[tuple[int, int, int, int, int, float]]]
    ] = []
    for instruction_index, instruction in enumerate(instructions):
        left_part = irreps_in1[instruction.input1_index]
        right_part = irreps_in2[instruction.input2_index]
        output_part = irreps_out[instruction.output_index]
        weight_slice = weight_slices[instruction_index]
        coefficients = _cg_array_data(
            instruction.ir_in1, instruction.ir_in2, instruction.ir_out
        )
        scale = instruction.normalization.scale()
        for c in range(instruction.ir_out.dim):
            terms = []
            for v in range(right_part.mul):
                for a in range(instruction.ir_in1.dim):
                    for b in range(instruction.ir_in2.dim):
                        coefficient = float(coefficients[a][b][c]) * scale
                        if coefficient == 0.0:
                            continue
                        terms.append(
                            (
                                left_blocks[instruction.input1_index] + a,
                                left_part.ir.dim,
                                right_blocks[instruction.input2_index]
                                + v * right_part.ir.dim
                                + b,
                                weight_slice.start + v,
                                right_part.mul,
                                coefficient,
                            )
                        )
            rows.append(
                (
                    output_blocks[instruction.output_index] + c,
                    output_part.ir.dim,
                    terms,
                )
            )
    width = max((len(row[2]) for row in rows), default=0)

    def integers(position: int):
        return [
            [term[position] for term in terms]
            + [0] * (width - len(terms))
            for _, _, terms in rows
        ]

    coefficients = [
        [term[5] for term in terms] + [0.0] * (width - len(terms))
        for _, _, terms in rows
    ]
    return ChannelTensorProductMetadata(
        left_offsets=mx.array(integers(0), dtype=mx.int32),
        left_strides=mx.array(integers(1), dtype=mx.int32),
        right_indices=mx.array(integers(2), dtype=mx.int32),
        weight_offsets=mx.array(integers(3), dtype=mx.int32),
        weight_strides=mx.array(integers(4), dtype=mx.int32),
        coefficients=mx.array(coefficients, dtype=mx.float32),
        output_offsets=mx.array([row[0] for row in rows], dtype=mx.int32),
        output_strides=mx.array([row[1] for row in rows], dtype=mx.int32),
        width=width,
        rows=len(rows),
        channels=channels,
        output_dim=irreps_out.dim,
    )


def _channel_kernels(
    input1_dim: int,
    input2_dim: int,
    output_dim: int,
    channels: int,
    rows: int,
    width: int,
):
    mx, _ = require_mlx()
    key = (input1_dim, input2_dim, output_dim, channels, rows, width)
    cached = _CHANNEL_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    inputs = [
        "left",
        "right",
        "weight",
        "left_offset",
        "left_stride",
        "right_index",
        "weight_offset",
        "weight_stride",
        "coefficient",
        "output_offset",
        "output_stride",
    ]
    source = f"""
uint tid = thread_position_in_grid.x;
uint channel = tid % {channels};
uint item = tid / {channels};
uint left_base = item * {input1_dim};
uint right_base = item * {input2_dim};
uint weight_base = weight_ndim == 1 ? 0 : item * weight_shape[1];
uint output_base = item * {output_dim};
for (uint row = 0; row < {rows}; ++row) {{
  float sum = 0.0f;
  for (uint entry = row * {width}; entry < (row + 1) * {width}; ++entry) {{
    float value = float(coefficient[entry]);
    if (value != 0.0f) {{
      value *= float(left[left_base + left_offset[entry] + channel * left_stride[entry]]);
      value *= float(right[right_base + right_index[entry]]);
      value *= float(weight[weight_base + weight_offset[entry] + channel * weight_stride[entry]]);
      sum += value;
    }}
  }}
  output[output_base + output_offset[row] + channel * output_stride[row]] = T(sum);
}}
"""
    forward = mx.fast.metal_kernel(
        name=f"e3nn_channel_tp_forward_{input1_dim}_{input2_dim}_{output_dim}_{channels}_{rows}_{width}",
        input_names=inputs,
        output_names=["output"],
        source=source,
    )
    backward_source = f"""
uint tid = thread_position_in_grid.x;
uint channel = tid % {channels};
uint item = tid / {channels};
uint left_base = item * {input1_dim};
uint right_base = item * {input2_dim};
uint weight_base = weight_ndim == 1 ? 0 : item * weight_shape[1];
uint output_base = item * {output_dim};
for (uint row = 0; row < {rows}; ++row) {{
  float cotangent_value = float(cotangent[output_base + output_offset[row] + channel * output_stride[row]]);
  for (uint entry = row * {width}; entry < (row + 1) * {width}; ++entry) {{
    float coefficient_value = float(coefficient[entry]);
    if (coefficient_value != 0.0f) {{
      uint left_index = left_base + left_offset[entry] + channel * left_stride[entry];
      uint right_location = right_base + right_index[entry];
      uint weight_index = weight_base + weight_offset[entry] + channel * weight_stride[entry];
      float left_value = float(left[left_index]);
      float right_value = float(right[right_location]);
      float weight_value = float(weight[weight_index]);
      float scaled = cotangent_value * coefficient_value;
      atomic_fetch_add_explicit(&left_gradient[left_index], scaled * right_value * weight_value, memory_order_relaxed);
      atomic_fetch_add_explicit(&right_gradient[right_location], scaled * left_value * weight_value, memory_order_relaxed);
      atomic_fetch_add_explicit(&weight_gradient[weight_index], scaled * left_value * right_value, memory_order_relaxed);
    }}
  }}
}}
"""
    backward = mx.fast.metal_kernel(
        name=f"e3nn_channel_tp_backward_{input1_dim}_{input2_dim}_{output_dim}_{channels}_{rows}_{width}",
        input_names=["left", "right", "weight", "cotangent", *inputs[3:]],
        output_names=["left_gradient", "right_gradient", "weight_gradient"],
        source=backward_source,
        atomic_outputs=True,
    )
    _CHANNEL_KERNEL_CACHE[key] = (forward, backward)
    return forward, backward


def make_channel_operation(
    metadata: ChannelTensorProductMetadata,
    input1_dim: int,
    input2_dim: int,
    general: Callable[..., Any],
):
    """Return the fused channel-local ``uvu`` Metal operation."""

    mx, _ = require_mlx()
    forward_kernel, backward_kernel = _channel_kernels(
        input1_dim,
        input2_dim,
        metadata.output_dim,
        metadata.channels,
        metadata.rows,
        metadata.width,
    )
    fixed = [
        metadata.left_offsets,
        metadata.left_strides,
        metadata.right_indices,
        metadata.weight_offsets,
        metadata.weight_strides,
        metadata.coefficients,
        metadata.output_offsets,
        metadata.output_strides,
    ]

    @mx.custom_function
    def differentiable_backward(left, right, weight, cotangent):
        return tuple(
            backward_kernel(
                inputs=[left, right, weight, cotangent, *fixed],
                template=[("T", left.dtype)],
                output_shapes=[left.shape, right.shape, weight.shape],
                output_dtypes=[left.dtype, right.dtype, weight.dtype],
                grid=(left.shape[0] * metadata.channels, 1, 1),
                threadgroup=(256, 1, 1),
                init_value=0,
            )
        )

    @differentiable_backward.vjp
    def differentiable_backward_vjp(primals, cotangents, _outputs):
        left, right, weight, cotangent = primals

        def general_backward(first, second, weight_value, output_cotangent):
            _, gradients = mx.vjp(
                general,
                (first, second, weight_value),
                (output_cotangent,),
            )
            return gradients

        _, gradients = mx.vjp(
            general_backward,
            (left, right, weight, cotangent),
            cotangents,
        )
        return gradients

    @mx.custom_function
    def operation(left, right, weight):
        return forward_kernel(
            inputs=[left, right, weight, *fixed],
            template=[("T", left.dtype)],
            output_shapes=[(left.shape[0], metadata.output_dim)],
            output_dtypes=[left.dtype],
            grid=(left.shape[0] * metadata.channels, 1, 1),
            threadgroup=(256, 1, 1),
            init_value=0,
        )[0]

    @operation.vjp
    def operation_vjp(primals, cotangent, _output):
        left, right, weight = primals
        return differentiable_backward(left, right, weight, cotangent)

    @operation.jvp
    def operation_jvp(primals, tangents):
        _, tangent = mx.jvp(general, primals, tangents)
        return tangent

    return operation


def _kernels(
    input1_dim: int,
    input2_dim: int,
    output_dim: int,
    width: int,
    weighted: bool,
):
    mx, _ = require_mlx()
    key = (input1_dim, input2_dim, output_dim, width, weighted)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    weight_forward = (
        "value *= float(weight[weight_base + weight_index[entry]]);"
        if weighted
        else ""
    )
    weight_base = "uint weight_base = weight_ndim == 1 ? 0 : item * weight_shape[1];" if weighted else ""
    forward_inputs = [
        "left",
        "right",
        *( ["weight"] if weighted else [] ),
        "left_index",
        "right_index",
        "weight_index",
        "coefficient",
    ]
    forward_source = f"""
uint tid = thread_position_in_grid.x;
uint output_component = tid % {output_dim};
uint item = tid / {output_dim};
uint left_base = item * {input1_dim};
uint right_base = item * {input2_dim};
{weight_base}
float sum = 0.0f;
for (uint entry = output_component * {width}; entry < (output_component + 1) * {width}; ++entry) {{
  float value = float(coefficient[entry]);
  if (value != 0.0f) {{
    value *= float(left[left_base + left_index[entry]]);
    value *= float(right[right_base + right_index[entry]]);
    {weight_forward}
    sum += value;
  }}
}}
output[tid] = T(sum);
"""
    forward = mx.fast.metal_kernel(
        name=f"e3nn_tp_forward_{input1_dim}_{input2_dim}_{output_dim}_{width}_{int(weighted)}",
        input_names=forward_inputs,
        output_names=["output"],
        source=forward_source,
    )

    weight_backward = ""
    backward_outputs = ["left_gradient", "right_gradient"]
    if weighted:
        backward_outputs.append("weight_gradient")
        weight_backward = """
    float weight_value = float(weight[weight_base + weight_index[entry]]);
    atomic_fetch_add_explicit(
        &weight_gradient[weight_base + weight_index[entry]],
        scaled * left_value * right_value,
        memory_order_relaxed);
"""
    else:
        weight_backward = "float weight_value = 1.0f;"
    backward_inputs = [
        "left",
        "right",
        "cotangent",
        *( ["weight"] if weighted else [] ),
        "left_index",
        "right_index",
        "weight_index",
        "coefficient",
    ]
    backward_source = f"""
uint tid = thread_position_in_grid.x;
uint output_component = tid % {output_dim};
uint item = tid / {output_dim};
uint left_base = item * {input1_dim};
uint right_base = item * {input2_dim};
{weight_base}
float cotangent_value = float(cotangent[tid]);
for (uint entry = output_component * {width}; entry < (output_component + 1) * {width}; ++entry) {{
  float coefficient_value = float(coefficient[entry]);
  if (coefficient_value != 0.0f) {{
    uint left_offset = left_base + left_index[entry];
    uint right_offset = right_base + right_index[entry];
    float left_value = float(left[left_offset]);
    float right_value = float(right[right_offset]);
    float scaled = cotangent_value * coefficient_value;
    {weight_backward}
    atomic_fetch_add_explicit(
        &left_gradient[left_offset], scaled * right_value * weight_value,
        memory_order_relaxed);
    atomic_fetch_add_explicit(
        &right_gradient[right_offset], scaled * left_value * weight_value,
        memory_order_relaxed);
  }}
}}
"""
    backward = mx.fast.metal_kernel(
        name=f"e3nn_tp_backward_{input1_dim}_{input2_dim}_{output_dim}_{width}_{int(weighted)}",
        input_names=backward_inputs,
        output_names=backward_outputs,
        source=backward_source,
        atomic_outputs=True,
    )
    _KERNEL_CACHE[key] = (forward, backward)
    return forward, backward


def make_operation(
    metadata: TensorProductKernelMetadata,
    input1_dim: int,
    input2_dim: int,
    general: Callable[..., Any],
):
    """Return a custom-differentiable generated tensor-product callable."""

    mx, _ = require_mlx()
    forward_kernel, backward_kernel = _kernels(
        input1_dim,
        input2_dim,
        metadata.output_dim,
        metadata.width,
        metadata.weighted,
    )
    fixed = [
        metadata.left_indices,
        metadata.right_indices,
        metadata.weight_indices,
        metadata.coefficients,
    ]

    def forward_inputs(left, right, weight):
        return [left, right, *([weight] if metadata.weighted else []), *fixed]

    def backward_inputs(left, right, cotangent, weight):
        return [
            left,
            right,
            cotangent,
            *([weight] if metadata.weighted else []),
            *fixed,
        ]

    @mx.custom_function
    def differentiable_backward(left, right, cotangent, weight):
        output_shapes = [left.shape, right.shape]
        output_dtypes = [left.dtype, right.dtype]
        if metadata.weighted:
            output_shapes.append(weight.shape)
            output_dtypes.append(weight.dtype)
        outputs = backward_kernel(
            inputs=backward_inputs(left, right, cotangent, weight),
            template=[("T", left.dtype)],
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            grid=(left.shape[0] * metadata.output_dim, 1, 1),
            threadgroup=(256, 1, 1),
            init_value=0,
        )
        if metadata.weighted:
            return outputs[0], outputs[1], outputs[2]
        return outputs[0], outputs[1], mx.zeros_like(weight)

    @differentiable_backward.vjp
    def differentiable_backward_vjp(primals, cotangents, _outputs):
        left, right, cotangent, weight = primals

        def general_backward(first, second, output_cotangent, weight_value):
            def apply(a, b, w):
                return general(a, b, w)

            _, gradients = mx.vjp(
                apply,
                (first, second, weight_value),
                (output_cotangent,),
            )
            return gradients

        def flattened(first, second, output_cotangent, weight_value):
            return general_backward(
                first, second, output_cotangent, weight_value
            )

        _, gradients = mx.vjp(
            flattened,
            (left, right, cotangent, weight),
            cotangents,
        )
        return gradients

    @mx.custom_function
    def operation(left, right, weight):
        return forward_kernel(
            inputs=forward_inputs(left, right, weight),
            template=[("T", left.dtype)],
            output_shapes=[(left.shape[0], metadata.output_dim)],
            output_dtypes=[left.dtype],
            grid=(left.shape[0] * metadata.output_dim, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]

    @operation.vjp
    def operation_vjp(primals, cotangent, _output):
        left, right, weight = primals
        gradients = differentiable_backward(left, right, cotangent, weight)
        return gradients

    @operation.jvp
    def operation_jvp(primals, tangents):
        _, tangent = mx.jvp(general, primals, tangents)
        return tangent

    return operation
