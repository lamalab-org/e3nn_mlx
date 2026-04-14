from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray
from e3nn_mlx.nn_linear import Linear
from e3nn_mlx.profiling import compare_eager_and_compiled


def main() -> None:
    linear = Linear("16x0e + 16x1o", "32x0e + 16x1o")
    array = IrrepsArray("16x0e + 16x1o", mlx_backend.asarray([[0.1] * (16 + 48)] * 2048))
    result = compare_eager_and_compiled(
        "linear_blockwise",
        lambda x: linear._apply_arrays(x, linear.weight, linear.bias),
        array.array,
        iters=20,
    )
    print(result)


if __name__ == "__main__":
    main()
