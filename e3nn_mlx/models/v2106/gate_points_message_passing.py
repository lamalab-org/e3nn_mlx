"""Gated modular message passing for v2106 point models."""

from __future__ import annotations

from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from ...compat import mlx_module_base, require_mlx
from ...irreps_array import IrrepsArray
from ...nn_gate import Gate
from .points_convolution import Convolution, _silu


def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    left = Irreps(irreps_in1).simplify()
    right = Irreps(irreps_in2).simplify()
    target = Irrep.parse(ir_out)
    return any(target in (part1.ir * part2.ir) for part1 in left for part2 in right)


class Compose(mlx_module_base()):
    def __init__(self, first, second) -> None:
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = first.irreps_in
        self.irreps_out = second.irreps_out

    def __call__(self, *inputs):
        return self.second(self.first(*inputs))


class MessagePassing(mlx_module_base()):
    """Sequence of v2106 convolutions and parity-aware gates."""

    def __init__(
        self,
        irreps_node_sequence,
        irreps_node_attr,
        irreps_edge_attr,
        fc_neurons,
        num_neighbors: float,
        *,
        use_custom_kernel: bool = True,
    ) -> None:
        super().__init__()
        requested_sequence = tuple(
            Irreps(irreps).remove_zero_multiplicities() for irreps in irreps_node_sequence
        )
        if len(requested_sequence) < 2:
            raise ValueError("irreps_node_sequence must contain input and output irreps")
        self.irreps_node_attr = Irreps(irreps_node_attr).remove_zero_multiplicities()
        self.irreps_edge_attr = Irreps(irreps_edge_attr).remove_zero_multiplicities()
        self.fc_neurons = tuple(int(width) for width in fc_neurons)
        if not self.fc_neurons or any(width <= 0 for width in self.fc_neurons):
            raise ValueError("fc_neurons must contain positive widths")
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be positive")
        self.num_neighbors = float(num_neighbors)
        self.use_custom_kernel = bool(use_custom_kernel)
        self._config = {
            "irreps_node_sequence": requested_sequence,
            "irreps_node_attr": self.irreps_node_attr,
            "irreps_edge_attr": self.irreps_edge_attr,
            "fc_neurons": self.fc_neurons,
            "num_neighbors": self.num_neighbors,
            "use_custom_kernel": self.use_custom_kernel,
        }

        mx, _ = require_mlx()
        scalar_activations = {1: _silu, -1: mx.tanh}
        gate_activations = {1: mx.sigmoid, -1: mx.tanh}
        modules = []
        actual_sequence = [requested_sequence[0]]
        current = requested_sequence[0]
        for hidden in requested_sequence[1:-1]:
            scalar_irreps = Irreps(
                part
                for part in hidden
                if part.ir.l == 0 and tp_path_exists(current, self.irreps_edge_attr, part.ir)
            ).simplify()
            gated_irreps = Irreps(
                part
                for part in hidden
                if part.ir.l > 0 and tp_path_exists(current, self.irreps_edge_attr, part.ir)
            )
            if gated_irreps:
                if tp_path_exists(current, self.irreps_edge_attr, "0e"):
                    gate_irrep = Irrep("0e")
                elif tp_path_exists(current, self.irreps_edge_attr, "0o"):
                    gate_irrep = Irrep("0o")
                else:
                    raise ValueError(
                        f"{current} x {self.irreps_edge_attr} cannot produce scalar gates"
                    )
            else:
                gate_irrep = Irrep("0e")
            gate_irreps = Irreps(
                MulIrrep(part.mul, gate_irrep) for part in gated_irreps
            ).simplify()
            gate = Gate(
                scalar_irreps,
                [scalar_activations[part.ir.p] for part in scalar_irreps],
                gate_irreps,
                [gate_activations[part.ir.p] for part in gate_irreps],
                gated_irreps,
            )
            convolution = Convolution(
                current,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                self.fc_neurons,
                self.num_neighbors,
                use_custom_kernel=self.use_custom_kernel,
            )
            modules.append(Compose(convolution, gate))
            current = gate.irreps_out
            actual_sequence.append(current)

        output = requested_sequence[-1]
        modules.append(
            Convolution(
                current,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                output,
                self.fc_neurons,
                self.num_neighbors,
                use_custom_kernel=self.use_custom_kernel,
            )
        )
        actual_sequence.append(output)
        self.layers = modules
        self.irreps_node_sequence = tuple(actual_sequence)
        self.irreps_node_input = actual_sequence[0]
        self.irreps_node_output = actual_sequence[-1]
        self.irreps_in = self.irreps_node_input
        self.irreps_out = self.irreps_node_output

    def __repr__(self) -> str:
        return f"MessagePassing ({self.irreps_node_input} -> {self.irreps_node_output})"

    def __deepcopy__(self, memo):
        copied = type(self)(**self._config)
        copied.update(self.parameters())
        copied.train(self.training)
        memo[id(self)] = copied
        return copied

    def __call__(
        self,
        node_features: IrrepsArray,
        node_attr: IrrepsArray,
        edge_src,
        edge_dst,
        edge_attr: IrrepsArray,
        edge_scalars,
    ) -> IrrepsArray:
        if node_features.irreps != self.irreps_node_input:
            raise ValueError("node_features irreps do not match MessagePassing input")
        if node_attr.irreps != self.irreps_node_attr:
            raise ValueError("node_attr irreps do not match MessagePassing attributes")
        if edge_attr.irreps != self.irreps_edge_attr:
            raise ValueError("edge_attr irreps do not match MessagePassing edge attributes")
        for layer in self.layers:
            node_features = layer(
                node_features,
                node_attr,
                edge_src,
                edge_dst,
                edge_attr,
                edge_scalars,
            )
        return node_features

    def forward_arrays(
        self,
        node_features,
        node_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
    ):
        return self(
            IrrepsArray(self.irreps_node_input, node_features),
            IrrepsArray(self.irreps_node_attr, node_attr),
            edge_src,
            edge_dst,
            IrrepsArray(self.irreps_edge_attr, edge_attr),
            edge_scalars,
        ).array
