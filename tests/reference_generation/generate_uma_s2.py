"""Generate S2-grid fixtures for the UMA compatibility surface.

Deterministic and deliberately not imported by the normal test suite. Run it by
hand, in the project's pinned reference environment, when the contract changes::

    uv venv .venv-reference --python 3.12
    VIRTUAL_ENV=.venv-reference uv pip install \
        -r tests/reference_generation/requirements.txt
    .venv-reference/bin/python tests/reference_generation/generate_uma_s2.py

Those pins (e3nn==0.5.8, torch==2.7.1, numpy==2.3.1) are the same ones the
``[reference]`` extra declares, so the fixtures are reproducible from the
repository alone. The manifest records the versions actually used; regenerating
with anything else will show up there.

Two families are covered:

* the UMA grids, all ``integral`` normalization, where the release gate is
  agreement of ``dense_to`` and ``dense_from`` -- the matrices eSCN/UMA consumes;
* ``component`` and ``norm`` cases including ``lmax_in != lmax``, which pin the
  forward and inverse scaling *absolutely*. A round-trip test cannot do this: a
  matching error in synthesis and analysis cancels out in the round trip.
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

# (identifier, kind, kwargs) for the normalization modes UMA does not use but
# this implementation still serves. lmax_in != lmax exercises the bandwidth
# correction in _from_s2_factors, the easiest part to get wrong.
NORMALIZATION_CASES = [
    ("to_l3_component", "to", {"lmax": 3, "normalization": "component"}),
    ("to_l3_norm", "to", {"lmax": 3, "normalization": "norm"}),
    ("to_l2_integral", "to", {"lmax": 2, "normalization": "integral"}),
    ("from_l3_component", "from", {"lmax": 3, "normalization": "component"}),
    ("from_l2_in3_component", "from", {"lmax": 2, "lmax_in": 3, "normalization": "component"}),
    ("from_l2_in3_norm", "from", {"lmax": 2, "lmax_in": 3, "normalization": "norm"}),
    ("from_l3_in5_component", "from", {"lmax": 3, "lmax_in": 5, "normalization": "component"}),
]


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
        "normalization_cases": [],
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

    for identifier, kind, kwargs in NORMALIZATION_CASES:
        grid = o3.ToS2Grid(**kwargs) if kind == "to" else o3.FromS2Grid(**kwargs)
        sha = grid.sha.numpy()
        shb = grid.shb.numpy()
        arrays[f"{identifier}_sha"] = sha
        arrays[f"{identifier}_shb"] = shb
        arrays[f"{identifier}_dense"] = (
            np.einsum("mbi,am->bai", shb, sha)
            if kind == "to"
            else np.einsum("am,mbi->bai", sha, shb)
        )
        manifest["normalization_cases"].append(
            {
                "id": identifier,
                "kind": kind,
                "res_beta": int(grid.res_beta),
                "res_alpha": int(grid.res_alpha),
                **kwargs,
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT / "uma_s2_reference.npz", **arrays)
    (OUTPUT / "uma_s2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUTPUT / 'uma_s2_reference.npz'} ({len(arrays)} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
