# Migrating from e3nn/PyTorch

The high-level namespaces deliberately resemble upstream e3nn:

| e3nn/PyTorch | e3nn-mlx |
| --- | --- |
| `import torch` | `import mlx.core as mx` |
| `from e3nn import o3, nn` | `from e3nn_mlx import o3, nn` |
| `torch.Tensor` | `mx.array` |
| `loss.backward()` | `mx.value_and_grad(...)` |
| `torch.jit` | `mx.compile` |
| `torch_scatter.scatter` | `e3nn_mlx.scatter_sum` |

Most constructors preserve the upstream names and representation arguments:

```python
import mlx.core as mx
from e3nn_mlx import nn, o3

linear = o3.Linear("16x0e + 8x1o", "32x0e + 8x1o")
gate = nn.Gate(
    "16x0e", [mx.tanh],
    "8x0e", [mx.sigmoid],
    "8x1o",
)
```

## Moving Linear weights

Default `Linear` paths and flattened weights follow upstream's input-major,
then output-index order. A flat upstream weight array can be copied without
reordering when the input irreps, output irreps, and explicit instructions are
the same. Use `weight_views(..., yield_instruction=True)` on both sides of a
converter to verify every path shape and meaning.

For weights generated per sample or graph edge, construct the MLX layer with
`internal_weights=False, shared_weights=False` and pass an array shaped
`(..., weight_numel)`. See the [Linear guide](linear.md) for the complete
layout and normalization contract.

## Deliberate differences

- MLX evaluates lazily. Include `mx.eval(...)` before reading results and when
  timing work.
- `mx.compile` traces fixed array computation. Dynamic neighbor discovery is
  eager; construct edges once and use a model's `forward_with_edges` method in
  compiled training or inference.
- `IrrepsArray` is optional in `e3nn_mlx.o3` and `e3nn_mlx.nn`, but the original
  flat API continues to use it for explicit representation checking.
- Generated Metal kernels are selected automatically and remain subject to
  dtype, shape, and autodiff-transform boundaries.
- TorchScript-specific APIs do not apply to MLX.

For an operation-by-operation statement of supported behavior, see
the [compatibility and numerical conventions](../COMPATIBILITY.md) and
[high-level API notes](../HIGH_LEVEL_API.md).
