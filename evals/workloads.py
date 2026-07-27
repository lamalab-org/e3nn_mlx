"""Small, shared workload definitions for the three-way benchmark."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CASE_DESCRIPTIONS = {
    "spherical_harmonics": (
        "Component-normalized spherical harmonics for geometric edge vectors."
    ),
    "full_tensor_product": (
        "Unweighted FullTensorProduct, including every allowed output path."
    ),
    "fully_connected_tensor_product": (
        "Learned dense equivariant mixing with shared weights."
    ),
    "weighted_tensor_product_uvu": (
        "Per-item uvu tensor product with externally generated weights."
    ),
    "linear": (
        "Equivariant multiplicity mixing; a control without a custom kernel."
    ),
    "scatter_sum": (
        "Destination-indexed graph message aggregation."
    ),
    "gate_points_2102": (
        "End-to-end February 2021 gated point-cloud network on fixed topology."
    ),
    "v2106_simple_network": (
        "End-to-end v2106 geometric message-passing network on fixed topology."
    ),
    "v2106_attributed_network": (
        "End-to-end v2106 network with node and edge attributes on fixed topology."
    ),
}


PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "smoke": {
        "spherical_harmonics": {"items": 256, "lmax": 3},
        "full_tensor_product": {"items": 32, "mul": 2, "lmax": 2},
        "fully_connected_tensor_product": {
            "items": 32,
            "mul": 3,
            "lmax": 2,
        },
        "weighted_tensor_product_uvu": {
            "items": 128,
            "mul": 3,
            "lmax": 2,
        },
        "linear": {"items": 256, "mul": 4, "lmax": 3},
        "scatter_sum": {"items": 512, "nodes": 64, "width": 32},
        "gate_points_2102": {
            "nodes": 16,
            "neighbors": 4,
            "mul": 3,
            "layers": 1,
            "lmax": 2,
        },
        "v2106_simple_network": {
            "nodes": 16,
            "neighbors": 4,
            "mul": 3,
            "layers": 1,
            "lmax": 2,
        },
        "v2106_attributed_network": {
            "nodes": 16,
            "neighbors": 4,
            "mul": 3,
            "layers": 1,
            "lmax": 2,
        },
    },
    "full": {
        "spherical_harmonics": {"items": 262_144, "lmax": 4},
        "full_tensor_product": {"items": 512, "mul": 8, "lmax": 3},
        "fully_connected_tensor_product": {
            "items": 2_048,
            "mul": 16,
            "lmax": 3,
        },
        "weighted_tensor_product_uvu": {
            "items": 65_536,
            "mul": 24,
            "lmax": 3,
        },
        "linear": {"items": 65_536, "mul": 32, "lmax": 4},
        "scatter_sum": {"items": 262_144, "nodes": 16_384, "width": 256},
        "gate_points_2102": {
            "nodes": 1_024,
            "neighbors": 12,
            "mul": 12,
            "layers": 3,
            "lmax": 3,
        },
        "v2106_simple_network": {
            "nodes": 1_024,
            "neighbors": 12,
            "mul": 16,
            "layers": 3,
            "lmax": 3,
        },
        "v2106_attributed_network": {
            "nodes": 1_024,
            "neighbors": 12,
            "mul": 16,
            "layers": 3,
            "lmax": 3,
        },
    },
}


DEFAULT_TIMING = {
    "smoke": {"warmup": 1, "samples": 3},
    "full": {"warmup": 8, "samples": 30},
}


def case_names() -> tuple[str, ...]:
    return tuple(CASE_DESCRIPTIONS)


def get_workloads(
    preset: str,
    selected: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    requested = list(PRESETS[preset]) if not selected else selected
    unknown = sorted(set(requested) - set(CASE_DESCRIPTIONS))
    if unknown:
        raise ValueError(f"unknown cases: {', '.join(unknown)}")
    return {name: deepcopy(PRESETS[preset][name]) for name in requested}


def spherical_irreps(mul: int, lmax: int) -> str:
    return " + ".join(
        f"{mul}x{degree}{'e' if degree % 2 == 0 else 'o'}"
        for degree in range(lmax + 1)
    )


def ring_edges(nodes: int, neighbors: int) -> tuple[list[int], list[int]]:
    if nodes <= 0 or neighbors <= 0 or neighbors >= nodes:
        raise ValueError("nodes must be positive and 0 < neighbors < nodes")
    source = []
    destination = []
    for node in range(nodes):
        for offset in range(1, neighbors + 1):
            source.append(node)
            destination.append((node + offset) % nodes)
    return source, destination
