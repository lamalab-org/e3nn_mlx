"""MLX port of e3nn's February 2021 gated point-cloud network."""

from __future__ import annotations

from math import cos, pi, sin, sqrt
from typing import Any

from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from ..compat import mlx_module_base, require_mlx
from ..graph import radius_graph, scatter_sum
from ..irreps_array import IrrepsArray
from ..nn_extract import ExtractIr
from ..nn_fc import FullyConnectedNet
from ..nn_gate import Gate
from ..ops_sh import spherical_harmonics
from ..ops_tp import FullyConnectedTensorProduct, TensorProduct
from ..radial import smooth_cutoff, soft_one_hot_linspace


def _silu(values):
    mx, _ = require_mlx()
    return values * mx.sigmoid(values)


def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    left = Irreps(irreps_in1).simplify()
    right = Irreps(irreps_in2).simplify()
    target = Irrep.parse(ir_out)
    return any(target in (part1.ir * part2.ir) for part1 in left for part2 in right)


def _as_irreps_array(value, irreps: Irreps, name: str) -> IrrepsArray:
    if isinstance(value, IrrepsArray):
        if value.irreps != irreps:
            raise ValueError(f"{name} irreps {value.irreps} do not match {irreps}")
        return value
    return IrrepsArray(irreps, value)


class Convolution(mlx_module_base()):
    """Equivariant edge convolution with radial TensorProduct weights."""

    def __init__(
        self,
        irreps_in,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_out,
        number_of_edge_features: int,
        radial_layers: int,
        radial_neurons: int,
        num_neighbors: float,
    ) -> None:
        super().__init__()
        if number_of_edge_features <= 0:
            raise ValueError("number_of_edge_features must be positive")
        if radial_layers < 0 or radial_neurons <= 0:
            raise ValueError("radial_layers must be non-negative and radial_neurons positive")
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be positive")
        self.irreps_in = Irreps(irreps_in).remove_zero_multiplicities()
        self.irreps_node_attr = Irreps(irreps_node_attr).remove_zero_multiplicities()
        self.irreps_edge_attr = Irreps(irreps_edge_attr).remove_zero_multiplicities()
        self.irreps_out = Irreps(irreps_out).remove_zero_multiplicities()
        self.number_of_edge_features = int(number_of_edge_features)
        self.num_neighbors = float(num_neighbors)

        self.sc = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_node_attr,
            self.irreps_out,
            use_custom_kernel=False,
        )
        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_node_attr,
            self.irreps_in,
            use_custom_kernel=False,
        )

        middle_parts = []
        instructions = []
        for input_index, input_part in enumerate(self.irreps_in):
            for edge_index, edge_part in enumerate(self.irreps_edge_attr):
                for output_irrep in input_part.ir * edge_part.ir:
                    if output_irrep in self.irreps_out:
                        output_index = len(middle_parts)
                        middle_parts.append(MulIrrep(input_part.mul, output_irrep))
                        instructions.append(
                            (input_index, edge_index, output_index, "uvu", True)
                        )
        sorted_middle = Irreps(middle_parts).sort()
        remapped = [
            (input_index, edge_index, sorted_middle.p[output_index], mode, train)
            for input_index, edge_index, output_index, mode, train in instructions
        ]
        self.irreps_mid_execution = sorted_middle.irreps
        if not self.irreps_mid_execution:
            raise ValueError(
                "input and edge irreps have no TensorProduct paths to the requested output"
            )
        self.irreps_mid = self.irreps_mid_execution.simplify()
        self.tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            self.irreps_mid_execution,
            remapped,
            internal_weights=False,
            shared_weights=False,
            use_custom_kernel=False,
        )
        widths = [self.number_of_edge_features]
        widths.extend([int(radial_neurons)] * int(radial_layers))
        widths.append(self.tp.weight_numel)
        self.fc = FullyConnectedNet(widths, _silu)
        self.lin2 = FullyConnectedTensorProduct(
            self.irreps_mid,
            self.irreps_node_attr,
            self.irreps_out,
            use_custom_kernel=False,
        )

    def __repr__(self) -> str:
        return f"Convolution ({self.irreps_in} -> {self.irreps_out})"

    def __call__(
        self,
        node_input: IrrepsArray,
        node_attr: IrrepsArray,
        edge_src,
        edge_dst,
        edge_attr: IrrepsArray,
        edge_features,
    ) -> IrrepsArray:
        if node_input.irreps != self.irreps_in:
            raise ValueError("node_input irreps do not match Convolution.irreps_in")
        if node_attr.irreps != self.irreps_node_attr:
            raise ValueError("node_attr irreps do not match Convolution.irreps_node_attr")
        if edge_attr.irreps != self.irreps_edge_attr:
            raise ValueError("edge_attr irreps do not match Convolution.irreps_edge_attr")
        if len(node_input.leading_shape) != 1 or len(node_attr.leading_shape) != 1:
            raise ValueError("node inputs and attributes must have shape (nodes, irreps.dim)")
        if node_input.shape[0] != node_attr.shape[0]:
            raise ValueError("node_input and node_attr must contain the same number of nodes")
        if edge_src.ndim != 1 or edge_dst.ndim != 1 or edge_src.shape != edge_dst.shape:
            raise ValueError("edge_src and edge_dst must be one-dimensional arrays of equal shape")
        mx, _ = require_mlx()
        if not mx.issubdtype(edge_src.dtype, mx.integer) or not mx.issubdtype(
            edge_dst.dtype, mx.integer
        ):
            raise TypeError("edge_src and edge_dst must have integer dtype")
        if edge_attr.shape[0] != edge_src.shape[0] or edge_features.shape[0] != edge_src.shape[0]:
            raise ValueError("edge arrays must contain the same number of edges")
        if edge_features.ndim != 2 or edge_features.shape[1] != self.number_of_edge_features:
            raise ValueError(
                f"edge_features must have shape (edges, {self.number_of_edge_features})"
            )

        sc_input = IrrepsArray(self.sc.irreps_in1, node_input.array)
        sc_attr = IrrepsArray(self.sc.irreps_in2, node_attr.array)
        self_connection = self.sc(sc_input, sc_attr)
        lin1_input = IrrepsArray(self.lin1.irreps_in1, node_input.array)
        lin1_attr = IrrepsArray(self.lin1.irreps_in2, node_attr.array)
        transformed_nodes = self.lin1(lin1_input, lin1_attr)
        if edge_src.shape[0] == 0:
            aggregated = mx.zeros(
                (node_input.shape[0], self.irreps_mid_execution.dim),
                dtype=node_input.dtype,
            )
        else:
            weights = self.fc(edge_features)
            gathered_nodes = IrrepsArray(self.tp.irreps_in1, transformed_nodes.array[edge_src])
            tp_edge_attr = IrrepsArray(self.tp.irreps_in2, edge_attr.array)
            messages = self.tp(gathered_nodes, tp_edge_attr, weights)
            aggregated = scatter_sum(messages.array, edge_dst, node_input.shape[0])
        aggregated = aggregated / sqrt(self.num_neighbors)
        middle = IrrepsArray(self.irreps_mid, aggregated)
        lin2_middle = IrrepsArray(self.lin2.irreps_in1, middle.array)
        lin2_attr = IrrepsArray(self.lin2.irreps_in2, node_attr.array)
        convolved = self.lin2(lin2_middle, lin2_attr)

        mask = self.sc.output_mask.astype(convolved.array.dtype)
        convolution_scale = (1.0 - mask) + cos(pi / 8) * mask
        output = sin(pi / 8) * self_connection.array + convolution_scale * convolved.array
        return IrrepsArray(self.irreps_out, output)

    def forward_arrays(
        self,
        node_input,
        node_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_features,
    ):
        return self(
            IrrepsArray(self.irreps_in, node_input),
            IrrepsArray(self.irreps_node_attr, node_attr),
            edge_src,
            edge_dst,
            IrrepsArray(self.irreps_edge_attr, edge_attr),
            edge_features,
        ).array


class Compose(mlx_module_base()):
    def __init__(self, first, second) -> None:
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = first.irreps_in
        self.irreps_out = second.irreps_out

    def __call__(self, *inputs):
        return self.second(self.first(*inputs))


class Network(mlx_module_base()):
    """Gated E(3)-equivariant network operating on one or more point graphs."""

    def __init__(
        self,
        irreps_in,
        irreps_hidden,
        irreps_out,
        irreps_node_attr,
        irreps_edge_attr,
        layers: int,
        max_radius: float,
        number_of_basis: int,
        radial_layers: int,
        radial_neurons: int,
        num_neighbors: float,
        num_nodes: float,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()
        if layers < 0:
            raise ValueError("layers must be non-negative")
        if max_radius <= 0 or number_of_basis <= 0 or num_nodes <= 0:
            raise ValueError("max_radius, number_of_basis, and num_nodes must be positive")
        self.max_radius = float(max_radius)
        self.number_of_basis = int(number_of_basis)
        self.num_neighbors = float(num_neighbors)
        self.num_nodes = float(num_nodes)
        self.reduce_output = bool(reduce_output)
        self.input_has_node_in = irreps_in is not None
        self.input_has_node_attr = irreps_node_attr is not None
        self.irreps_in = Irreps(irreps_in).simplify() if irreps_in is not None else None
        self.irreps_hidden = Irreps(irreps_hidden).simplify()
        self.irreps_out = Irreps(irreps_out).simplify()
        self.irreps_node_attr = (
            Irreps(irreps_node_attr).simplify()
            if irreps_node_attr is not None
            else Irreps("0e")
        )
        self.irreps_edge_attr = Irreps(irreps_edge_attr).simplify()
        self._config = {
            "irreps_in": self.irreps_in,
            "irreps_hidden": self.irreps_hidden,
            "irreps_out": self.irreps_out,
            "irreps_node_attr": self.irreps_node_attr if self.input_has_node_attr else None,
            "irreps_edge_attr": self.irreps_edge_attr,
            "layers": int(layers),
            "max_radius": self.max_radius,
            "number_of_basis": self.number_of_basis,
            "radial_layers": int(radial_layers),
            "radial_neurons": int(radial_neurons),
            "num_neighbors": self.num_neighbors,
            "num_nodes": self.num_nodes,
            "reduce_output": self.reduce_output,
        }
        self.ext_z = ExtractIr(self.irreps_node_attr, "0e")
        edge_feature_count = self.number_of_basis + 2 * self.irreps_node_attr.count("0e")
        current = self.irreps_in if self.irreps_in is not None else Irreps("0e")
        modules = []

        mx, _ = require_mlx()
        scalar_activations = {1: _silu, -1: mx.tanh}
        gate_activations = {1: mx.sigmoid, -1: mx.tanh}
        for _ in range(layers):
            scalar_irreps = Irreps(
                part
                for part in self.irreps_hidden
                if part.ir.l == 0 and tp_path_exists(current, self.irreps_edge_attr, part.ir)
            )
            gated_irreps = Irreps(
                part
                for part in self.irreps_hidden
                if part.ir.l > 0 and tp_path_exists(current, self.irreps_edge_attr, part.ir)
            )
            if gated_irreps:
                if tp_path_exists(current, self.irreps_edge_attr, "0e"):
                    gate_irrep = Irrep("0e")
                elif tp_path_exists(current, self.irreps_edge_attr, "0o"):
                    gate_irrep = Irrep("0o")
                else:
                    raise ValueError("TensorProduct paths cannot produce scalar gates")
            else:
                gate_irrep = Irrep("0e")
            gate_irreps = Irreps(MulIrrep(part.mul, gate_irrep) for part in gated_irreps)
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
                edge_feature_count,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
            modules.append(Compose(convolution, gate))
            current = gate.irreps_out
        modules.append(
            Convolution(
                current,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                edge_feature_count,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
        )
        self.layers = modules

    def __repr__(self) -> str:
        return f"Network ({self.irreps_in} -> {self.irreps_out}, layers={len(self.layers)})"

    def __deepcopy__(self, memo):
        config = dict(self._config)
        config["reduce_output"] = self.reduce_output
        copied = type(self)(**config)
        copied.update(self.parameters())
        copied.train(self.training)
        memo[id(self)] = copied
        return copied

    def _default_features(self, positions, width: int):
        mx, _ = require_mlx()
        return mx.ones((positions.shape[0], width), dtype=positions.dtype)

    def forward_with_edges(
        self,
        positions,
        node_input,
        node_attr,
        batch,
        edge_src,
        edge_dst,
        *,
        num_graphs: int,
    ) -> IrrepsArray:
        mx, _ = require_mlx()
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (nodes, 3)")
        if batch.ndim != 1 or batch.shape[0] != positions.shape[0]:
            raise ValueError("batch must have shape (nodes,)")
        if not mx.issubdtype(batch.dtype, mx.integer):
            raise TypeError("batch must have an integer dtype")
        if num_graphs < 0:
            raise ValueError("num_graphs must be non-negative")

        current_irreps = self.irreps_in if self.irreps_in is not None else Irreps("0e")
        features = _as_irreps_array(node_input, current_irreps, "node_input")
        attributes = _as_irreps_array(node_attr, self.irreps_node_attr, "node_attr")
        if features.shape[0] != positions.shape[0] or attributes.shape[0] != positions.shape[0]:
            raise ValueError("positions, node_input, and node_attr must contain the same nodes")
        edge_vectors = positions[edge_src] - positions[edge_dst]
        edge_harmonics = spherical_harmonics(
            self.irreps_edge_attr,
            edge_vectors,
            normalize=True,
            normalization="component",
            # Keep the fully transformable recurrence inside end-to-end model
            # graphs; the generated SH kernel is benchmarked/selected at the
            # operator boundary and currently has a CustomKernel JVP limit.
            use_custom_kernel=False,
        )
        edge_lengths = mx.sqrt(mx.sum(edge_vectors * edge_vectors, axis=-1))
        radial = soft_one_hot_linspace(
            edge_lengths,
            0.0,
            self.max_radius,
            self.number_of_basis,
            "gaussian",
            False,
        ) * sqrt(self.number_of_basis)
        edge_attributes = IrrepsArray(
            self.irreps_edge_attr,
            smooth_cutoff(edge_lengths / self.max_radius)[:, None] * edge_harmonics,
        )
        scalar_attributes = self.ext_z(attributes).array
        edge_features = mx.concatenate(
            [radial, scalar_attributes[edge_src], scalar_attributes[edge_dst]], axis=-1
        )
        for layer in self.layers:
            features = layer(
                features,
                attributes,
                edge_src,
                edge_dst,
                edge_attributes,
                edge_features,
            )
        if self.reduce_output:
            reduced = scatter_sum(features.array, batch, num_graphs) / sqrt(self.num_nodes)
            return IrrepsArray(self.irreps_out, reduced)
        return features

    def __call__(self, data: dict[str, Any]) -> IrrepsArray:
        mx, _ = require_mlx()
        if "pos" not in data:
            raise KeyError("data must contain 'pos'")
        positions = data["pos"]
        node_count = positions.shape[0]
        batch = data.get("batch", mx.zeros((node_count,), dtype=mx.int32))
        edges = radius_graph(positions, self.max_radius, batch)
        if self.input_has_node_in:
            if "x" not in data:
                raise KeyError("data must contain 'x' when irreps_in is specified")
            node_input = data["x"]
        else:
            node_input = self._default_features(positions, 1)
        if self.input_has_node_attr:
            if "z" not in data:
                raise KeyError("data must contain 'z' when irreps_node_attr is specified")
            node_attr = data["z"]
        else:
            node_attr = self._default_features(positions, 1)
        num_graphs = 0 if node_count == 0 else int(mx.max(batch).item()) + 1
        return self.forward_with_edges(
            positions,
            node_input,
            node_attr,
            batch,
            edges[0],
            edges[1],
            num_graphs=num_graphs,
        )
