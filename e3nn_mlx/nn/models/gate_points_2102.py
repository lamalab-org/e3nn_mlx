"""Upstream-shaped adapter for the February 2021 gated point model."""

from __future__ import annotations

from ..._high_level import ensure_irreps_array, unwrap_irreps_arrays
from ...models.gate_points_2102 import (
    Compose as _Compose,
    Convolution as _Convolution,
    Network as _Network,
    tp_path_exists,
)


class Compose(_Compose):
    forward = _Compose.__call__


class Convolution(_Convolution):
    def __call__(
        self,
        node_input,
        node_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_features,
    ):
        features, raw_features = ensure_irreps_array(node_input, self.irreps_in)
        attributes, raw_attributes = ensure_irreps_array(
            node_attr, self.irreps_node_attr
        )
        edges, raw_edges = ensure_irreps_array(edge_attr, self.irreps_edge_attr)
        output = super().__call__(
            features,
            attributes,
            edge_src,
            edge_dst,
            edges,
            edge_features,
        )
        return unwrap_irreps_arrays(
            output, raw_features and raw_attributes and raw_edges
        )

    forward = __call__


class Network(_Network):
    def __call__(self, data):
        raw = not hasattr(data.get("x"), "irreps")
        output = super().__call__(data)
        return unwrap_irreps_arrays(output, raw)

    forward = __call__


__all__ = ["Compose", "Convolution", "Network", "tp_path_exists"]
