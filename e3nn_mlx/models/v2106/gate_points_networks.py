"""End-to-end modular v2106 point-cloud networks."""

from __future__ import annotations

from math import sqrt
from typing import Any

from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from ...compat import mlx_module_base, require_mlx
from ...graph import radius_graph, scatter_sum
from ...irreps_array import IrrepsArray
from ...ops_sh import spherical_harmonics
from ...radial import soft_one_hot_linspace
from .gate_points_message_passing import MessagePassing


def _as_irreps_array(value, irreps: Irreps, name: str) -> IrrepsArray:
    if isinstance(value, IrrepsArray):
        if value.irreps != irreps:
            raise ValueError(f"{name} irreps {value.irreps} do not match {irreps}")
        return value
    return IrrepsArray(irreps, value)


def _hidden_irreps(mul: int, lmax: int) -> Irreps:
    return Irreps(
        MulIrrep(mul, Irrep(l, parity))
        for l in range(lmax + 1)
        for parity in (-1, 1)
    )


def _validate_graph_arrays(positions, batch, edge_src, edge_dst) -> None:
    mx, _ = require_mlx()
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (nodes, 3)")
    if batch.ndim != 1 or batch.shape[0] != positions.shape[0]:
        raise ValueError("batch must have shape (nodes,)")
    if not mx.issubdtype(batch.dtype, mx.integer):
        raise TypeError("batch must have integer dtype")
    if edge_src.ndim != 1 or edge_dst.ndim != 1 or edge_src.shape != edge_dst.shape:
        raise ValueError("edge_src and edge_dst must be equal one-dimensional arrays")
    if not mx.issubdtype(edge_src.dtype, mx.integer) or not mx.issubdtype(
        edge_dst.dtype, mx.integer
    ):
        raise TypeError("edge_src and edge_dst must have integer dtype")


class _PointNetworkBase(mlx_module_base()):
    number_of_basis = 10

    def _edge_geometry(self, positions, edge_src, edge_dst):
        mx, _ = require_mlx()
        edge_vectors = positions[edge_src] - positions[edge_dst]
        edge_harmonics = spherical_harmonics(
            self.irreps_sh,
            edge_vectors,
            normalize=True,
            normalization="component",
        )
        edge_lengths = mx.sqrt(mx.sum(edge_vectors * edge_vectors, axis=-1))
        edge_scalars = soft_one_hot_linspace(
            edge_lengths,
            0.0,
            self.max_radius,
            self.number_of_basis,
            "smooth_finite",
            True,
        ) * sqrt(self.number_of_basis)
        return edge_harmonics, edge_scalars

    def _finish(self, output: IrrepsArray, batch, num_graphs: int) -> IrrepsArray:
        if num_graphs < 0:
            raise ValueError("num_graphs must be non-negative")
        if self.pool_nodes:
            pooled = scatter_sum(output.array, batch, num_graphs) / sqrt(self.num_nodes)
            return IrrepsArray(self.irreps_node_output, pooled)
        return output

    def __deepcopy__(self, memo):
        config = dict(self._config)
        config["pool_nodes"] = self.pool_nodes
        copied = type(self)(**config)
        copied.update(self.parameters())
        copied.train(self.training)
        memo[id(self)] = copied
        return copied


class SimpleNetwork(_PointNetworkBase):
    """v2106 point network that derives all geometric edge attributes from positions."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
        max_radius: float,
        num_neighbors: float,
        num_nodes: float,
        mul: int = 50,
        layers: int = 3,
        lmax: int = 2,
        pool_nodes: bool = True,
    ) -> None:
        super().__init__()
        if max_radius <= 0 or num_neighbors <= 0 or num_nodes <= 0:
            raise ValueError("max_radius, num_neighbors, and num_nodes must be positive")
        if mul <= 0 or layers < 0 or lmax < 0:
            raise ValueError("mul must be positive; layers and lmax must be non-negative")
        self.max_radius = float(max_radius)
        self.num_nodes = float(num_nodes)
        self.pool_nodes = bool(pool_nodes)
        self.lmax = int(lmax)
        self.irreps_sh = Irreps.spherical_harmonics(self.lmax)
        self._config = {
            "irreps_in": Irreps(irreps_in).remove_zero_multiplicities(),
            "irreps_out": Irreps(irreps_out).remove_zero_multiplicities(),
            "max_radius": self.max_radius,
            "num_neighbors": float(num_neighbors),
            "num_nodes": self.num_nodes,
            "mul": int(mul),
            "layers": int(layers),
            "lmax": self.lmax,
            "pool_nodes": self.pool_nodes,
        }
        hidden = _hidden_irreps(int(mul), self.lmax)
        self.mp = MessagePassing(
            [
                self._config["irreps_in"],
                *([hidden] * int(layers)),
                self._config["irreps_out"],
            ],
            "0e",
            self.irreps_sh,
            [self.number_of_basis, 100],
            num_neighbors,
        )
        self.irreps_in = self.mp.irreps_node_input
        self.irreps_out = self.mp.irreps_node_output
        self.irreps_node_input = self.irreps_in
        self.irreps_node_attr = self.mp.irreps_node_attr
        self.irreps_node_output = self.irreps_out
        self.irreps_edge_attr = self.mp.irreps_edge_attr

    def __repr__(self) -> str:
        return (
            f"SimpleNetwork ({self.irreps_in} -> {self.irreps_out}, "
            f"layers={len(self.mp.layers)})"
        )

    def preprocess(self, data: dict[str, Any]):
        mx, _ = require_mlx()
        if "pos" not in data:
            raise KeyError("data must contain 'pos'")
        if "x" not in data:
            raise KeyError("data must contain 'x'")
        positions = data["pos"]
        batch = data.get("batch", mx.zeros((positions.shape[0],), dtype=mx.int32))
        edges = radius_graph(positions, self.max_radius, batch)
        edge_vectors = positions[edges[0]] - positions[edges[1]]
        return batch, data["x"], edges[0], edges[1], edge_vectors

    def forward_with_edges(
        self,
        positions,
        node_input,
        batch,
        edge_src,
        edge_dst,
        *,
        num_graphs: int,
    ) -> IrrepsArray:
        mx, _ = require_mlx()
        _validate_graph_arrays(positions, batch, edge_src, edge_dst)
        features = _as_irreps_array(node_input, self.irreps_node_input, "node_input")
        if features.shape[0] != positions.shape[0]:
            raise ValueError("positions and node_input must contain the same nodes")
        edge_harmonics, edge_scalars = self._edge_geometry(positions, edge_src, edge_dst)
        node_attr = IrrepsArray(
            self.irreps_node_attr,
            mx.ones((positions.shape[0], 1), dtype=features.dtype),
        )
        output = self.mp(
            features,
            node_attr,
            edge_src,
            edge_dst,
            IrrepsArray(self.irreps_sh, edge_harmonics),
            edge_scalars,
        )
        return self._finish(output, batch, num_graphs)

    def __call__(self, data: dict[str, Any]) -> IrrepsArray:
        mx, _ = require_mlx()
        batch, node_input, edge_src, edge_dst, _ = self.preprocess(data)
        num_graphs = 0 if batch.shape[0] == 0 else int(mx.max(batch).item()) + 1
        return self.forward_with_edges(
            data["pos"],
            node_input,
            batch,
            edge_src,
            edge_dst,
            num_graphs=num_graphs,
        )


class NetworkForAGraphWithAttributes(_PointNetworkBase):
    """v2106 point network with caller-provided node and edge attributes."""

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        max_radius: float,
        num_neighbors: float,
        num_nodes: float,
        mul: int = 50,
        layers: int = 3,
        lmax: int = 2,
        pool_nodes: bool = True,
    ) -> None:
        super().__init__()
        if max_radius <= 0 or num_neighbors <= 0 or num_nodes <= 0:
            raise ValueError("max_radius, num_neighbors, and num_nodes must be positive")
        if mul <= 0 or layers < 0 or lmax < 0:
            raise ValueError("mul must be positive; layers and lmax must be non-negative")
        self.max_radius = float(max_radius)
        self.num_nodes = float(num_nodes)
        self.pool_nodes = bool(pool_nodes)
        self.lmax = int(lmax)
        self.irreps_edge_attr = Irreps(irreps_edge_attr).remove_zero_multiplicities()
        self.irreps_sh = Irreps.spherical_harmonics(self.lmax)
        self.irreps_edge_attr_combined = self.irreps_edge_attr + self.irreps_sh
        self._config = {
            "irreps_node_input": Irreps(irreps_node_input).remove_zero_multiplicities(),
            "irreps_node_attr": Irreps(irreps_node_attr).remove_zero_multiplicities(),
            "irreps_edge_attr": self.irreps_edge_attr,
            "irreps_node_output": Irreps(irreps_node_output).remove_zero_multiplicities(),
            "max_radius": self.max_radius,
            "num_neighbors": float(num_neighbors),
            "num_nodes": self.num_nodes,
            "mul": int(mul),
            "layers": int(layers),
            "lmax": self.lmax,
            "pool_nodes": self.pool_nodes,
        }
        hidden = _hidden_irreps(int(mul), self.lmax)
        self.mp = MessagePassing(
            [
                self._config["irreps_node_input"],
                *([hidden] * int(layers)),
                self._config["irreps_node_output"],
            ],
            self._config["irreps_node_attr"],
            self.irreps_edge_attr_combined,
            [self.number_of_basis, 100],
            num_neighbors,
        )
        self.irreps_node_input = self.mp.irreps_node_input
        self.irreps_node_attr = self.mp.irreps_node_attr
        self.irreps_node_output = self.mp.irreps_node_output
        self.irreps_in = self.irreps_node_input
        self.irreps_out = self.irreps_node_output

    def __repr__(self) -> str:
        return (
            "NetworkForAGraphWithAttributes "
            f"({self.irreps_node_input} -> {self.irreps_node_output}, layers={len(self.mp.layers)})"
        )

    def preprocess(self, data: dict[str, Any]):
        mx, _ = require_mlx()
        if "pos" not in data:
            raise KeyError("data must contain 'pos'")
        if "node_attr" not in data or "edge_attr" not in data:
            raise KeyError("data must contain 'node_attr' and 'edge_attr'")
        if "x" in data:
            node_input = data["x"]
        elif "node_input" in data:
            node_input = data["node_input"]
        else:
            raise KeyError("data must contain 'x' or 'node_input'")
        positions = data["pos"]
        batch = data.get("batch", mx.zeros((positions.shape[0],), dtype=mx.int32))
        if "edge_index" in data:
            edge_index = data["edge_index"]
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape (2, edges)")
        else:
            edge_index = radius_graph(positions, self.max_radius, batch)
        edge_vectors = positions[edge_index[0]] - positions[edge_index[1]]
        return (
            batch,
            node_input,
            data["node_attr"],
            data["edge_attr"],
            edge_index[0],
            edge_index[1],
            edge_vectors,
        )

    def forward_with_edges(
        self,
        positions,
        node_input,
        node_attr,
        edge_attr,
        batch,
        edge_src,
        edge_dst,
        *,
        num_graphs: int,
    ) -> IrrepsArray:
        mx, _ = require_mlx()
        _validate_graph_arrays(positions, batch, edge_src, edge_dst)
        features = _as_irreps_array(node_input, self.irreps_node_input, "node_input")
        attributes = _as_irreps_array(node_attr, self.irreps_node_attr, "node_attr")
        user_edge_attr = _as_irreps_array(edge_attr, self.irreps_edge_attr, "edge_attr")
        if features.shape[0] != positions.shape[0] or attributes.shape[0] != positions.shape[0]:
            raise ValueError("positions, node_input, and node_attr must contain the same nodes")
        if user_edge_attr.shape[0] != edge_src.shape[0]:
            raise ValueError("edge_attr must contain one row per edge")
        edge_harmonics, edge_scalars = self._edge_geometry(positions, edge_src, edge_dst)
        combined = mx.concatenate([user_edge_attr.array, edge_harmonics], axis=-1)
        output = self.mp(
            features,
            attributes,
            edge_src,
            edge_dst,
            IrrepsArray(self.irreps_edge_attr_combined, combined),
            edge_scalars,
        )
        return self._finish(output, batch, num_graphs)

    def __call__(self, data: dict[str, Any]) -> IrrepsArray:
        mx, _ = require_mlx()
        values = self.preprocess(data)
        batch = values[0]
        num_graphs = 0 if batch.shape[0] == 0 else int(mx.max(batch).item()) + 1
        return self.forward_with_edges(
            data["pos"],
            values[1],
            values[2],
            values[3],
            batch,
            values[4],
            values[5],
            num_graphs=num_graphs,
        )


__all__ = ["NetworkForAGraphWithAttributes", "SimpleNetwork"]
