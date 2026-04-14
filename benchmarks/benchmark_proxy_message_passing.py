from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray
from e3nn_mlx.nn_linear import Linear
from e3nn_mlx.ops_tp import tensor_product
from e3nn_mlx.profiling import compare_eager_and_compiled


def main() -> None:
    linear_node = Linear("8x0e + 8x1o", "8x0e + 8x1o")
    linear_edge = Linear("1o", "1o")
    node = IrrepsArray("8x0e + 8x1o", mlx_backend.asarray([[0.1] * (8 + 24)] * 1024))
    edge = IrrepsArray("1o", mlx_backend.asarray([[0.2, 0.3, 0.4]] * 1024))

    def proxy_layer(x, e):
        x_proj = IrrepsArray(linear_node.irreps_out, linear_node._apply_arrays(x, linear_node.weight, linear_node.bias))
        e_proj = IrrepsArray(linear_edge.irreps_out, linear_edge._apply_arrays(e, linear_edge.weight, linear_edge.bias))
        return tensor_product(IrrepsArray("1o", x_proj.chunk_arrays()[1][..., :3]), e_proj).array

    result = compare_eager_and_compiled("proxy_message_passing", proxy_layer, node.array, edge.array, iters=20)
    print(result)


if __name__ == "__main__":
    main()
