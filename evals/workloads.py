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
