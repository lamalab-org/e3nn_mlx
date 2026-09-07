"""Generate S2-grid fixtures for the UMA compatibility surface.

Deterministic and deliberately not imported by the normal test suite: run it
by hand against the pinned upstream environment when the contract changes.

    python tests/reference_generation/generate_uma_s2.py

The release gate is agreement of ``dense_to`` and ``dense_from``, the matrices
eSCN/UMA actually consumes.
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

# (lmax, mmax) pairs covering an mmax == lmax grid and two undersampled ones.
CASES = [(2, 2), (4, 2), (6, 2)]


def grid_resolution(lmax: int, mmax: int) -> tuple[int, int]:
    res_beta = 2 * (lmax + 1)
    res_alpha = 2 * (mmax + 1) + 1 if lmax == mmax else 2 * mmax + 1
    return res_beta, res_alpha


def main() -> int:
    torch.set_default_dtype(torch.float64)
    arrays: dict[str, np.ndarray] = {}
    manifest = {
        "schema": 1,
        "e3nn": e3nn.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "normalization": "integral",
        "cases": [],
    }

    for lmax, mmax in CASES:
        res_beta, res_alpha = grid_resolution(lmax, mmax)
        to_grid = o3.ToS2Grid(lmax, (res_beta, res_alpha), normalization="integral")
        from_grid = o3.FromS2Grid((res_beta, res_alpha), lmax, normalization="integral")

        key = f"{lmax}_{mmax}"
        to_sha = to_grid.sha.numpy()
        to_shb = to_grid.shb.numpy()
        from_sha = from_grid.sha.numpy()
        from_shb = from_grid.shb.numpy()

        arrays[f"{key}_to_sha"] = to_sha
        arrays[f"{key}_to_shb"] = to_shb
        arrays[f"{key}_from_sha"] = from_sha
        arrays[f"{key}_from_shb"] = from_shb
        arrays[f"{key}_dense_to"] = np.einsum("mbi,am->bai", to_shb, to_sha)
        arrays[f"{key}_dense_from"] = np.einsum("am,mbi->bai", from_sha, from_shb)

        manifest["cases"].append(
            {
                "lmax": lmax,
                "mmax": mmax,
                "res_beta": res_beta,
                "res_alpha": res_alpha,
                "num_coefficients": (lmax + 1) ** 2,
                "num_m_modes": 2 * lmax + 1,
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT / "uma_s2_reference.npz", **arrays)
    (OUTPUT / "uma_s2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUTPUT / 'uma_s2_reference.npz'} ({len(arrays)} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
