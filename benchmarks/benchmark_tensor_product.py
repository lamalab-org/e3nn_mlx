from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray
from e3nn_mlx.ops_tp import compile_tensor_product, tensor_product, tensor_product_plan
from e3nn_mlx.profiling import compare_eager_and_compiled


def main() -> None:
    left = IrrepsArray("1o", mlx_backend.asarray([[0.2, 0.3, 0.4]] * 4096))
    right = IrrepsArray("1o", mlx_backend.asarray([[0.5, 0.7, 0.9]] * 4096))
    plan = tensor_product_plan("1o", "1o")
    eager = lambda a, b: tensor_product(IrrepsArray("1o", a), IrrepsArray("1o", b)).array
    compiled = compile_tensor_product(plan)
    result = compare_eager_and_compiled("tensor_product_1x1", eager, left.array, right.array, iters=20)
    mlx_backend._require().eval(compiled(left.array, right.array))
    print(result)


if __name__ == "__main__":
    main()
