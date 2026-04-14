from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.ops_sh import spherical_harmonics
from e3nn_mlx.profiling import compare_eager_and_compiled


def main() -> None:
    vectors = mlx_backend.asarray([[0.2, 0.3, 0.4]] * 4096)
    result = compare_eager_and_compiled(
        "spherical_harmonics_l2",
        lambda x: spherical_harmonics(2, x, normalize=True, normalization="component"),
        vectors,
        iters=20,
    )
    print(result)


if __name__ == "__main__":
    main()
