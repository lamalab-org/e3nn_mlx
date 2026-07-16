"""Upstream-shaped adapters for complete modular v2106 networks."""

from __future__ import annotations

from ...._high_level import unwrap_irreps_arrays
from ....irreps_array import IrrepsArray
from ....models.v2106.gate_points_networks import (
    NetworkForAGraphWithAttributes as _NetworkForAGraphWithAttributes,
    SimpleNetwork as _SimpleNetwork,
)


class SimpleNetwork(_SimpleNetwork):
    def __call__(self, data):
        raw = not isinstance(data.get("x"), IrrepsArray)
        return unwrap_irreps_arrays(super().__call__(data), raw)

    forward = __call__


class NetworkForAGraphWithAttributes(_NetworkForAGraphWithAttributes):
    def __call__(self, data):
        node_input = data.get("x", data.get("node_input"))
        raw = not isinstance(node_input, IrrepsArray)
        return unwrap_irreps_arrays(super().__call__(data), raw)

    forward = __call__


__all__ = ["NetworkForAGraphWithAttributes", "SimpleNetwork"]
