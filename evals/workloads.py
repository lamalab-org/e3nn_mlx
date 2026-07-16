"""Shared workload definitions for equivalent MLX and PyTorch benchmarks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CASE_DESCRIPTIONS = {
    "spherical_harmonics": (
        "Component-normalized spherical harmonics for edge directions; stresses polynomial "
        "recurrences and small tensor kernels used by every geometric network."
    ),
    "full_tensor_product": (
        "Unweighted full tensor product; stresses Clebsch-Gordan contractions and exposes "
        "the cost of materializing all representation-product channels."
    ),
    "fully_connected_tensor_product": (
        "Learned fully connected tensor product; representative of the dense equivariant "
        "mixing used inside convolutions."
    ),
    "linear": (
        "Blockwise equivariant Linear layer; measures multiplicity mixing without geometric "
        "coupling and is often bandwidth or launch-overhead limited."
    ),
    "v2106_convolution": (
        "One modular v2106 convolution on a fixed sparse graph, including radial MLP, tensor "
        "product, scatter aggregation, residual path, and learned alpha mixing."
    ),
    "v2106_message_passing": (
        "A gated stack of v2106 convolutions on fixed topology; captures repeated sparse "
        "aggregation, nonlinear gates, and intermediate representations."
    ),
    "v2106_network": (
        "End-to-end attributed v2106 network with fixed topology, position-derived spherical "
        "harmonics and radial basis, message passing, and graph pooling."
    ),
}


PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "smoke": {
        "spherical_harmonics": {"items": 512, "lmax": 3},
        "full_tensor_product": {"items": 64, "mul": 2, "lmax": 2},
        "fully_connected_tensor_product": {
            "items": 128,
            "mul": 4,
            "lmax": 2,
        },
        "linear": {"items": 512, "mul": 8, "lmax": 3},
        "v2106_convolution": {
            "nodes": 64,
            "neighbors": 4,
            "mul": 4,
            "lmax": 2,
            "radial": 10,
            "radial_hidden": 16,
        },
        "v2106_message_passing": {
            "nodes": 48,
            "neighbors": 4,
            "mul": 3,
            "lmax": 2,
            "layers": 2,
            "radial": 10,
            "radial_hidden": 16,
        },
        "v2106_network": {
            "nodes": 48,
            "neighbors": 4,
            "mul": 3,
            "lmax": 2,
            "layers": 2,
        },
    },
    "medium": {
        "spherical_harmonics": {"items": 65_536, "lmax": 6},
        "full_tensor_product": {"items": 2_048, "mul": 4, "lmax": 3},
        "fully_connected_tensor_product": {
            "items": 4_096,
            "mul": 24,
            "lmax": 3,
        },
        "linear": {"items": 16_384, "mul": 32, "lmax": 4},
        "v2106_convolution": {
            "nodes": 2_048,
            "neighbors": 16,
            "mul": 24,
            "lmax": 3,
            "radial": 10,
            "radial_hidden": 64,
        },
        "v2106_message_passing": {
            "nodes": 1_024,
            "neighbors": 16,
            "mul": 20,
            "lmax": 3,
            "layers": 3,
            "radial": 10,
            "radial_hidden": 64,
        },
        "v2106_network": {
            "nodes": 1_024,
            "neighbors": 16,
            "mul": 20,
            "lmax": 3,
            "layers": 3,
        },
    },
    "large": {
        "spherical_harmonics": {"items": 1_048_576, "lmax": 8},
        "full_tensor_product": {"items": 4_096, "mul": 8, "lmax": 3},
        "fully_connected_tensor_product": {
            "items": 8_192,
            "mul": 32,
            "lmax": 4,
        },
        "linear": {"items": 131_072, "mul": 64, "lmax": 5},
        "v2106_convolution": {
            "nodes": 8_192,
            "neighbors": 24,
            "mul": 32,
            "lmax": 4,
            "radial": 10,
            "radial_hidden": 128,
        },
        "v2106_message_passing": {
            "nodes": 4_096,
            "neighbors": 24,
            "mul": 24,
            "lmax": 3,
            "layers": 4,
            "radial": 10,
            "radial_hidden": 128,
        },
        "v2106_network": {
            "nodes": 4_096,
            "neighbors": 24,
            "mul": 24,
            "lmax": 3,
            "layers": 4,
        },
    },
}


DEFAULT_TIMING = {
    "smoke": {"warmup": 2, "samples": 7, "inner_repeats": 1},
    "medium": {"warmup": 5, "samples": 15, "inner_repeats": 1},
    "large": {"warmup": 5, "samples": 20, "inner_repeats": 1},
}


def case_names() -> tuple[str, ...]:
    return tuple(CASE_DESCRIPTIONS)


def get_workloads(preset: str, selected: list[str] | None = None) -> dict[str, dict[str, Any]]:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
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
