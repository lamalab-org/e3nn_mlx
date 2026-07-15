"""Generate deterministic numerical fixtures from pinned upstream e3nn.

This script is deliberately not imported by the normal test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import e3nn
from e3nn import o3
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reference_data"
SEED = 20260716


def _versions() -> dict[str, str | int]:
    return {
        "schema": 1,
        "seed": SEED,
        "e3nn": e3nn.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "lmax_guaranteed": 6,
        "lmax_tested": 8,
    }


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float64)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    triples: list[tuple[int, int, int]] = []
    arrays: dict[str, np.ndarray] = {}
    for l1 in range(7):
        for l2 in range(7):
            for l3 in range(abs(l1 - l2), min(l1 + l2, 6) + 1):
                triples.append((l1, l2, l3))
                arrays[f"w3j_{l1}_{l2}_{l3}"] = o3.wigner_3j(l1, l2, l3).numpy()

    angles = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, -0.4, 0.7], [-1.1, 0.8, 2.2]],
        dtype=torch.float64,
    )
    arrays["angles"] = angles.numpy()
    for l in range(9):
        ir = o3.Irrep(l, 1)
        arrays[f"wigner_d_{l}"] = ir.D_from_angles(angles[:, 0], angles[:, 1], angles[:, 2]).numpy()

    vectors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.2, -0.5, 0.7], [-0.3, 0.4, 0.1]],
        dtype=torch.float64,
    )
    arrays["vectors"] = vectors.numpy()
    for normalization in ("component", "norm", "integral"):
        for l in range(7):
            arrays[f"sh_{normalization}_{l}"] = o3.spherical_harmonics(
                l, vectors, normalize=True, normalization=normalization
            ).numpy()

    np.savez_compressed(OUTPUT / "e3nn_p0_reference.npz", **arrays)
    manifest = _versions() | {"wigner_3j_triples": triples}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

