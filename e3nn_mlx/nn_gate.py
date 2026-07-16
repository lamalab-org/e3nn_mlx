"""MLX Gate module."""

from __future__ import annotations

from typing import Any

from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from .compat import mlx_module_base, require_mlx
from .irreps_array import IrrepsArray
from .nn_activation import Activation
from .ops_tp import ElementwiseTensorProduct


class _Sortcut(mlx_module_base()):
    """Sort a direct sum and recover its original representation groups."""

    def __init__(self, *irreps_outs: Irreps | str) -> None:
        super().__init__()
        self.irreps_outs = tuple(Irreps(value).simplify() for value in irreps_outs)
        tagged = [
            (group_index, block_index, part)
            for group_index, irreps in enumerate(self.irreps_outs)
            for block_index, part in enumerate(irreps)
        ]
        tagged.sort(key=lambda item: item[2].ir)
        self._routes = tuple((group_index, block_index) for group_index, block_index, _ in tagged)
        self.irreps_in = Irreps(part for _, _, part in tagged)

    def __call__(self, features: IrrepsArray) -> tuple[IrrepsArray, ...]:
        if features.irreps != self.irreps_in:
            raise ValueError("input irreps do not match _Sortcut.irreps_in")
        mx, _ = require_mlx()
        grouped: list[list[Any | None]] = [[None for _ in irreps] for irreps in self.irreps_outs]
        for chunk, (group_index, block_index) in zip(features.chunk_arrays(), self._routes, strict=True):
            grouped[group_index][block_index] = chunk
        outputs = []
        for irreps, chunks in zip(self.irreps_outs, grouped, strict=True):
            selected = [chunk for chunk in chunks if chunk is not None]
            array = mx.concatenate(selected, axis=-1) if selected else mx.zeros((*features.leading_shape, 0), dtype=features.dtype)
            outputs.append(IrrepsArray(irreps, array))
        return tuple(outputs)


class Gate(mlx_module_base()):
    def __init__(
        self,
        irreps_scalars: Irreps | str,
        *args,
        scalar_activation=None,
        gate_activation=None,
        even_scalar_activation=None,
        odd_scalar_activation=None,
        even_gate_activation=None,
        odd_gate_activation=None,
    ) -> None:
        super().__init__()
        if len(args) == 2:
            irreps_gates, irreps_gated = args
            act_scalars = act_gates = None
            self._upstream_api = False
        elif len(args) == 4:
            act_scalars, irreps_gates, act_gates, irreps_gated = args
            self._upstream_api = True
        else:
            raise TypeError(
                "Gate expects (scalars, gates, gated) or (scalars, scalar_acts, gates, gate_acts, gated)"
            )

        self.irreps_scalars = Irreps(irreps_scalars).remove_zero_multiplicities()
        self.irreps_gates = Irreps(irreps_gates).remove_zero_multiplicities()
        self.irreps_gated = Irreps(irreps_gated).remove_zero_multiplicities()
        if any(part.ir.l != 0 for part in self.irreps_scalars):
            raise ValueError("irreps_scalars must be scalar; only scalar irreps are allowed")
        if any(part.ir.l != 0 for part in self.irreps_gates):
            raise ValueError("irreps_gates must be scalar; only scalar irreps are allowed")
        if self.irreps_gates.num_irreps != self.irreps_gated.num_irreps:
            raise ValueError("the number of gate scalars must match the gated multiplicity")

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

        if self._upstream_api:
            activation_irreps_scalars = self.irreps_scalars
            activation_irreps_gates = self.irreps_gates
            self.sc = _Sortcut(self.irreps_scalars, self.irreps_gates, self.irreps_gated)
            self.irreps_scalars, self.irreps_gates, self.irreps_gated = self.sc.irreps_outs
            self.irreps_in = self.sc.irreps_in
            self.act_scalars = Activation(activation_irreps_scalars, act_scalars)
            self.act_gates = Activation(activation_irreps_gates, act_gates)
            self.mul = ElementwiseTensorProduct(
                self.irreps_gated,
                self.act_gates.irreps_out,
                compile_left_right=False,
            )
            self.irreps_out = self.act_scalars.irreps_out + self.mul.irreps_out
        else:
            if len(self.irreps_gates) != len(self.irreps_gated):
                raise ValueError("legacy Gate requires the same number of gate and gated blocks")
            for gate_part, gated_part in zip(self.irreps_gates, self.irreps_gated, strict=True):
                if gate_part.mul != gated_part.mul:
                    raise ValueError("each gate block must match the gated multiplicity")
            self.irreps_in = self.irreps_scalars + self.irreps_gates + self.irreps_gated
            gated_outputs = Irreps(
                MulIrrep(gated_part.mul, Irrep(gated_part.ir.l, gated_part.ir.p * gate_part.ir.p))
                for gate_part, gated_part in zip(self.irreps_gates, self.irreps_gated, strict=True)
            )
            self.irreps_out = self.irreps_scalars + gated_outputs

    def __repr__(self) -> str:
        return f"Gate ({self.irreps_in} -> {self.irreps_out})"

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
        if self._upstream_api:
            scalars, gates, gated = self.sc(array)
            scalars = IrrepsArray(self.act_scalars.irreps_in, scalars.array)
            scalars = self.act_scalars(scalars)
            if not self.irreps_gates:
                return scalars
            gates = IrrepsArray(self.act_gates.irreps_in, gates.array)
            gates = self.act_gates(gates)
            gated_outputs = []
            gate_cursor = 0
            for part, chunk in zip(
                gated.irreps, gated.chunk_arrays(), strict=True
            ):
                gate_values = gates.array[
                    ..., gate_cursor : gate_cursor + part.mul
                ].reshape(*array.leading_shape, part.mul, 1)
                gated_values = chunk.reshape(
                    *array.leading_shape, part.mul, part.ir.dim
                )
                gated_outputs.append(
                    (gate_values * gated_values).reshape(
                        *array.leading_shape, part.dim
                    )
                )
                gate_cursor += part.mul
            product = (
                mx.concatenate(gated_outputs, axis=-1)
                if gated_outputs
                else mx.zeros((*array.leading_shape, 0), dtype=array.dtype)
            )
            return IrrepsArray(
                self.irreps_out, mx.concatenate([scalars.array, product], axis=-1)
            )

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
            outputs.append((gates * gated).reshape(*array.leading_shape, gated_part.dim))
        result = mx.concatenate(outputs, axis=-1) if outputs else mx.zeros((*array.leading_shape, 0), dtype=array.dtype)
        return IrrepsArray(self.irreps_out, result)
