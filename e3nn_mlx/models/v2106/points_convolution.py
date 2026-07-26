"""Modular v2106 equivariant point convolution."""

from __future__ import annotations

from math import sqrt

from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from ...compat import mlx_module_base, require_mlx
from ...graph import scatter_sum
from ...irreps_array import IrrepsArray
from ...nn_fc import FullyConnectedNet
from ...ops_tp import FullyConnectedTensorProduct, TensorProduct


def _silu(values):
    mx, _ = require_mlx()
    return values * mx.sigmoid(values)


class Convolution(mlx_module_base()):
    """v2106 convolution with a learned scalar residual mixing factor."""

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        num_neighbors: float,
        *,
        use_custom_kernel: bool = True,
    ) -> None:
        super().__init__()
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be positive")
        self.irreps_node_input = Irreps(irreps_node_input).remove_zero_multiplicities()
        self.irreps_node_attr = Irreps(irreps_node_attr).remove_zero_multiplicities()
        self.irreps_edge_attr = Irreps(irreps_edge_attr).remove_zero_multiplicities()
        self.irreps_node_output = Irreps(irreps_node_output).remove_zero_multiplicities()
        self.irreps_in = self.irreps_node_input
        self.irreps_out = self.irreps_node_output
        self.fc_neurons = tuple(int(width) for width in fc_neurons)
        if not self.fc_neurons or any(width <= 0 for width in self.fc_neurons):
            raise ValueError("fc_neurons must contain positive input and hidden widths")
        self.num_neighbors = float(num_neighbors)
        self.use_custom_kernel = bool(use_custom_kernel)

        self.sc = FullyConnectedTensorProduct(
            self.irreps_node_input,
            self.irreps_node_attr,
            self.irreps_node_output,
            use_custom_kernel=self.use_custom_kernel,
        )
        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_node_input,
            self.irreps_node_attr,
            self.irreps_node_input,
            use_custom_kernel=self.use_custom_kernel,
        )

        middle_parts = []
        instructions = []
        invariant_scalar = Irrep("0e")
        for input_index, input_part in enumerate(self.irreps_node_input):
            for edge_index, edge_part in enumerate(self.irreps_edge_attr):
                for output_irrep in input_part.ir * edge_part.ir:
                    if output_irrep in self.irreps_node_output or output_irrep == invariant_scalar:
                        output_index = len(middle_parts)
                        middle_parts.append(MulIrrep(input_part.mul, output_irrep))
                        instructions.append(
                            (input_index, edge_index, output_index, "uvu", True)
                        )
        sorted_middle = Irreps(middle_parts).sort()
        self.irreps_mid_execution = sorted_middle.irreps
        if not self.irreps_mid_execution:
            raise ValueError("TensorProduct produces no paths for the requested output")
        remapped = [
            (input_index, edge_index, sorted_middle.p[output_index], mode, train)
            for input_index, edge_index, output_index, mode, train in instructions
        ]
        self.irreps_mid = self.irreps_mid_execution.simplify()
        self.tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            self.irreps_mid_execution,
            remapped,
            internal_weights=False,
            shared_weights=False,
            use_custom_kernel=self.use_custom_kernel,
        )
        self.fc = FullyConnectedNet([*self.fc_neurons, self.tp.weight_numel], _silu)
        self.lin2 = FullyConnectedTensorProduct(
            self.irreps_mid,
            self.irreps_node_attr,
            self.irreps_node_output,
            use_custom_kernel=self.use_custom_kernel,
        )
        self.alpha = FullyConnectedTensorProduct(
            self.irreps_mid,
            self.irreps_node_attr,
            "0e",
            use_custom_kernel=self.use_custom_kernel,
        )
        mx, _ = require_mlx()
        if self.alpha.weight is None or not bool(mx.all(self.alpha.output_mask)):
            raise ValueError("intermediate and node-attribute irreps cannot produce alpha scalars")
        self.alpha.update({"weight": mx.zeros_like(self.alpha.weight)})

    def __repr__(self) -> str:
        return f"Convolution ({self.irreps_node_input} -> {self.irreps_node_output})"

    def __call__(
        self,
        node_input: IrrepsArray,
        node_attr: IrrepsArray,
        edge_src,
        edge_dst,
        edge_attr: IrrepsArray,
        edge_scalars,
    ) -> IrrepsArray:
        mx, _ = require_mlx()
        if node_input.irreps != self.irreps_node_input:
            raise ValueError("node_input irreps do not match")
        if node_attr.irreps != self.irreps_node_attr:
            raise ValueError("node_attr irreps do not match")
        if edge_attr.irreps != self.irreps_edge_attr:
            raise ValueError("edge_attr irreps do not match")
        if len(node_input.leading_shape) != 1 or len(node_attr.leading_shape) != 1:
            raise ValueError("node inputs and attributes must have shape (nodes, irreps.dim)")
        if node_input.shape[0] != node_attr.shape[0]:
            raise ValueError("node_input and node_attr must contain the same nodes")
        if edge_src.ndim != 1 or edge_dst.ndim != 1 or edge_src.shape != edge_dst.shape:
            raise ValueError("edge_src and edge_dst must be equal one-dimensional arrays")
        if not mx.issubdtype(edge_src.dtype, mx.integer) or not mx.issubdtype(
            edge_dst.dtype, mx.integer
        ):
            raise TypeError("edge_src and edge_dst must have integer dtype")
        if edge_attr.shape[0] != edge_src.shape[0] or edge_scalars.shape[0] != edge_src.shape[0]:
            raise ValueError("edge arrays must contain the same number of edges")
        if edge_scalars.ndim != 2 or edge_scalars.shape[1] != self.fc_neurons[0]:
            raise ValueError(f"edge_scalars must have shape (edges, {self.fc_neurons[0]})")

        sc_input = IrrepsArray(self.sc.irreps_in1, node_input.array)
        sc_attr = IrrepsArray(self.sc.irreps_in2, node_attr.array)
        self_connection = self.sc(sc_input, sc_attr)
        lin1_input = IrrepsArray(self.lin1.irreps_in1, node_input.array)
        lin1_attr = IrrepsArray(self.lin1.irreps_in2, node_attr.array)
        transformed = self.lin1(lin1_input, lin1_attr)

        if edge_src.shape[0] == 0:
            aggregated = mx.zeros(
                (node_input.shape[0], self.irreps_mid_execution.dim),
                dtype=node_input.dtype,
            )
        else:
            weights = self.fc(edge_scalars)
            gathered = IrrepsArray(self.tp.irreps_in1, transformed.array[edge_src])
            tp_edge_attr = IrrepsArray(self.tp.irreps_in2, edge_attr.array)
            messages = self.tp(gathered, tp_edge_attr, weights)
            aggregated = scatter_sum(
                messages.array,
                edge_dst,
                node_input.shape[0],
                use_custom_kernel=self.use_custom_kernel,
            )
        aggregated = aggregated / sqrt(self.num_neighbors)

        middle = IrrepsArray(self.irreps_mid, aggregated)
        lin2_middle = IrrepsArray(self.lin2.irreps_in1, middle.array)
        lin2_attr = IrrepsArray(self.lin2.irreps_in2, node_attr.array)
        convolved = self.lin2(lin2_middle, lin2_attr)
        alpha_middle = IrrepsArray(self.alpha.irreps_in1, middle.array)
        alpha_attr = IrrepsArray(self.alpha.irreps_in2, node_attr.array)
        alpha = self.alpha(alpha_middle, alpha_attr).array

        mask = self.sc.output_mask.astype(convolved.array.dtype)
        mixing = (1.0 - mask) + alpha * mask
        return IrrepsArray(
            self.irreps_node_output,
            self_connection.array + mixing * convolved.array,
        )

    def forward_arrays(
        self,
        node_input,
        node_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
    ):
        return self(
            IrrepsArray(self.irreps_node_input, node_input),
            IrrepsArray(self.irreps_node_attr, node_attr),
            edge_src,
            edge_dst,
            IrrepsArray(self.irreps_edge_attr, edge_attr),
            edge_scalars,
        ).array
