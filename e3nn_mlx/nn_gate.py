"""MLX Gate module."""

from __future__ import annotations

from e3nn_core.irreps import Irreps

from .compat import require_mlx
from .irreps_array import IrrepsArray


class Gate:
    def __init__(
        self,
        irreps_scalars: Irreps | str,
        irreps_gates: Irreps | str,
        irreps_gated: Irreps | str,
        *,
        scalar_activation=None,
        gate_activation=None,
    ) -> None:
        self.irreps_scalars = Irreps(irreps_scalars).remove_zero_multiplicities()
        self.irreps_gates = Irreps(irreps_gates).remove_zero_multiplicities()
        self.irreps_gated = Irreps(irreps_gated).remove_zero_multiplicities()
        if len(self.irreps_gates) != len(self.irreps_gated):
            raise ValueError("irreps_gates and irreps_gated must have the same number of blocks")
        for gate_part, gated_part in zip(self.irreps_gates, self.irreps_gated, strict=True):
            if gate_part.ir.l != 0 or gate_part.mul != gated_part.mul:
                raise ValueError("each gate block must be scalar and match the gated multiplicity")
        self.irreps_in = self.irreps_scalars + self.irreps_gates + self.irreps_gated
        self.irreps_out = self.irreps_scalars + self.irreps_gated
        self.scalar_activation = scalar_activation
        self.gate_activation = gate_activation

    def __call__(self, array: IrrepsArray) -> IrrepsArray:
        mx, _ = require_mlx()
        if array.irreps != self.irreps_in:
            raise ValueError("input irreps do not match Gate.irreps_in")
        scalar_activation = self.scalar_activation or mx.tanh
        gate_activation = self.gate_activation or mx.sigmoid
        chunks = list(array.chunk_arrays())
        scalar_count = len(self.irreps_scalars)
        gate_count = len(self.irreps_gates)
        scalar_chunks = chunks[:scalar_count]
        gate_chunks = chunks[scalar_count : scalar_count + gate_count]
        gated_chunks = chunks[scalar_count + gate_count :]

        outputs = [scalar_activation(chunk) for chunk in scalar_chunks]
        for gate_part, gated_part, gate_chunk, gated_chunk in zip(
            self.irreps_gates, self.irreps_gated, gate_chunks, gated_chunks, strict=True
        ):
            gates = gate_activation(gate_chunk).reshape(*array.leading_shape, gate_part.mul, 1)
            gated = gated_chunk.reshape(*array.leading_shape, gated_part.mul, gated_part.ir.dim)
            outputs.append((gates * gated).reshape(*array.leading_shape, -1))
        return IrrepsArray(self.irreps_out, mx.concatenate(outputs, axis=-1))
