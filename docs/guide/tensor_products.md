# Tensor products

Tensor products are the central learnable interaction in e3nn. They combine
two representations and project their outer product onto allowed output
irreps using Clebsch–Gordan coefficients.

## Standard products

{class}`~e3nn_mlx.o3.FullTensorProduct` emits every allowed output path and has
no learned weights:

```python
import mlx.core as mx
from e3nn_mlx import o3

tp = o3.FullTensorProduct("2x1o", "3x1o")
left = mx.random.normal((64, tp.irreps_in1.dim))
right = mx.random.normal((64, tp.irreps_in2.dim))
output = tp(left, right)
```

{class}`~e3nn_mlx.o3.FullyConnectedTensorProduct` connects all compatible
multiplicity channels with learned weights:

```python
tp = o3.FullyConnectedTensorProduct(
    "8x0e + 8x1o",
    "1x0e + 1x1o",
    "16x0e + 8x1o",
)
output = tp(left_features, edge_attributes)
```

Use {class}`~e3nn_mlx.o3.ElementwiseTensorProduct` for aligned channels and
{class}`~e3nn_mlx.o3.TensorSquare` when both inputs are the same value.

## General instructions

The general {class}`~e3nn_mlx.o3.TensorProduct` accepts an explicit output
representation and instruction list. Each instruction is
`(input1_index, input2_index, output_index, connection_mode, has_weight)`;
an optional sixth value scales the path before normalization. This interface is
powerful but low level—prefer a standard wrapper when it represents the desired
connectivity.

## Shared and per-sample weights

With `shared_weights=True`, weights have shape `(weight_numel,)`. With
`shared_weights=False`, external weights have shape
`(..., weight_numel)`, which is useful when a radial network predicts a tensor
product for every graph edge.

## Compilation and generated kernels

Compatible float32, rank-two inputs dispatch to specialized Metal kernels on
Apple silicon. Other shapes, dtypes, large dense contractions, and unsupported
instruction mixtures automatically use general MLX operations. Both paths
implement the same contraction and normalization conventions.
