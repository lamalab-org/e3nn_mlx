"""Upstream-shaped adapter for modular v2106 message passing."""

from __future__ import annotations

from ...._high_level import ensure_irreps_array, unwrap_irreps_arrays
from ....models.v2106.gate_points_message_passing import (
    Compose as _Compose,
    MessagePassing as _MessagePassing,
    tp_path_exists,
)


class Compose(_Compose):
    forward = _Compose.__call__


class MessagePassing(_MessagePassing):
    def __call__(
        self,
        node_features,
        node_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
    ):
        features, raw_features = ensure_irreps_array(
            node_features, self.irreps_node_input
        )
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
            edge_scalars,
        )
        return unwrap_irreps_arrays(
            output, raw_features and raw_attributes and raw_edges
        )

    forward = __call__


__all__ = ["Compose", "MessagePassing", "tp_path_exists"]
