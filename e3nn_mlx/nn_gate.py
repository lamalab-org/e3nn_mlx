"""MLX Gate module."""

from __future__ import annotations

from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from .compat import mlx_module_base, require_mlx
from .irreps_array import IrrepsArray


class Gate(mlx_module_base()):
    def __init__(
        self,
        irreps_scalars: Irreps | str,
        irreps_gates: Irreps | str,
        irreps_gated: Irreps | str,
        *,
        scalar_activation=None,
        gate_activation=None,
        even_scalar_activation=None,
        odd_scalar_activation=None,
        even_gate_activation=None,
        odd_gate_activation=None,
    ) -> None:
        super().__init__()
        self.irreps_scalars = Irreps(irreps_scalars).remove_zero_multiplicities()
        self.irreps_gates = Irreps(irreps_gates).remove_zero_multiplicities()
        self.irreps_gated = Irreps(irreps_gated).remove_zero_multiplicities()
        if len(self.irreps_gates) != len(self.irreps_gated):
            raise ValueError("irreps_gates and irreps_gated must have the same number of blocks")
        for gate_part, gated_part in zip(self.irreps_gates, self.irreps_gated, strict=True):
            if gate_part.ir.l != 0 or gate_part.mul != gated_part.mul:
                raise ValueError("each gate block must be scalar and match the gated multiplicity")
        self.irreps_in = self.irreps_scalars + self.irreps_gates + self.irreps_gated
        gated_outputs = Irreps(
            MulIrrep(gated_part.mul, Irrep(gated_part.ir.l, gated_part.ir.p * gate_part.ir.p))
            for gate_part, gated_part in zip(self.irreps_gates, self.irreps_gated, strict=True)
        )
        self.irreps_out = self.irreps_scalars + gated_outputs
        if scalar_activation is not None and (even_scalar_activation is not None or odd_scalar_activation is not None):
            raise ValueError("scalar_activation cannot be combined with parity-specific scalar activations")
        if gate_activation is not None and (even_gate_activation is not None or odd_gate_activation is not None):
            raise ValueError("gate_activation cannot be combined with parity-specific gate activations")
        self.scalar_activation = scalar_activation
        self.gate_activation = gate_activation
        self.even_scalar_activation = even_scalar_activation
        self.odd_scalar_activation = odd_scalar_activation
        self.even_gate_activation = even_gate_activation
        self.odd_gate_activation = odd_gate_activation

    def _scalar_activation_for(self, parity: int, mx):
        if self.scalar_activation is not None:
            return self.scalar_activation
        if parity == 1:
            return self.even_scalar_activation or mx.tanh
        return self.odd_scalar_activation or mx.tanh

    def _gate_activation_for(self, parity: int, mx):
        if self.gate_activation is not None:
            return self.gate_activation
        if parity == 1:
            return self.even_gate_activation or mx.sigmoid
        return self.odd_gate_activation or mx.tanh

    def __call__(self, array: IrrepsArray) -> IrrepsArray:
        mx, _ = require_mlx()
        if array.irreps != self.irreps_in:
            raise ValueError("input irreps do not match Gate.irreps_in")
        chunks = list(array.chunk_arrays())
        scalar_count = len(self.irreps_scalars)
        gate_count = len(self.irreps_gates)
        scalar_chunks = chunks[:scalar_count]
        gate_chunks = chunks[scalar_count : scalar_count + gate_count]
        gated_chunks = chunks[scalar_count + gate_count :]

        outputs = [
            self._scalar_activation_for(part.ir.p, mx)(chunk)
            for part, chunk in zip(self.irreps_scalars, scalar_chunks, strict=True)
        ]
        for gate_part, gated_part, gate_chunk, gated_chunk in zip(
            self.irreps_gates, self.irreps_gated, gate_chunks, gated_chunks, strict=True
        ):
            activation = self._gate_activation_for(gate_part.ir.p, mx)
            gates = activation(gate_chunk).reshape(*array.leading_shape, gate_part.mul, 1)
            gated = gated_chunk.reshape(*array.leading_shape, gated_part.mul, gated_part.ir.dim)
            outputs.append((gates * gated).reshape(*array.leading_shape, -1))
        if not outputs:
            result = mx.zeros((*array.leading_shape, 0), dtype=array.array.dtype)
        else:
            result = mx.concatenate(outputs, axis=-1)
        return IrrepsArray(self.irreps_out, result)
