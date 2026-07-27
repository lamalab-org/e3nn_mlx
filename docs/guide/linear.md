# Equivariant linear layers

{class}`~e3nn_mlx.o3.Linear` mixes multiplicity channels belonging to the same
irrep. It never connects different angular momenta or parities, so scalar,
vector, and higher-order channels retain their transformation laws.

```python
import mlx.core as mx
from e3nn_mlx import o3

linear = o3.Linear(
    "2x0e + 3x1o + 1x0e",
    "4x0e + 2x1o",
)
x = mx.random.normal((32, linear.irreps_in.dim))
y = linear(x)
```

## Instructions and flattened weights

An explicit instruction is `(input_index, output_index)`. The input and output
blocks must carry identical irreps. Without explicit instructions, `Linear`
enumerates compatible paths in input-major, then output-index order, matching
upstream e3nn.

Each instruction owns a weight matrix with shape
`(input_multiplicity, output_multiplicity)`. `weight_numel` is the sum of those
matrix sizes, and the flattened weight vector concatenates the matrices in
instruction order. Inspect the layout instead of manually reproducing slices:

```python
external = o3.Linear(
    "2x0e + 3x1o + 1x0e",
    "4x0e + 2x1o",
    internal_weights=False,
)
weights = mx.random.normal((external.weight_numel,))

for instruction_index, instruction, view in external.weight_views(
    weights, yield_instruction=True
):
    assert view.shape == instruction.path_shape

y = external(x, weights)
```

The order is part of checkpoint and externally generated weight semantics.
When converting weights from upstream e3nn, preserve the flat vector as-is
after verifying that the source and destination irreps and explicit
instructions are identical.

## Shared and per-sample weights

Shared weights have shape `(weight_numel,)`. Set `shared_weights=False` for
external weights with shape `(..., weight_numel)`; their leading dimensions
broadcast with the input leading dimensions.

```python
per_sample = o3.Linear(
    "2x0e + 3x1o",
    "4x0e + 2x1o",
    internal_weights=False,
    shared_weights=False,
)
batch = mx.random.normal((32, per_sample.irreps_in.dim))
weights = mx.random.normal((32, per_sample.weight_numel))
output = per_sample(batch, weights)
```

If `f_in` and `f_out` are supplied, the module also mixes explicit feature
channels and expects weights ending in
`(f_in, f_out, weight_numel)`.

## Normalization, bias, and differentiation

`path_normalization="element"` normalizes each output by the total number of
contributing input elements. `"path"` gives each path equal aggregate
normalization. The normalization factor is applied during the forward
contraction, so internal and external weights have identical semantics.

Bias is disabled by default and is allowed only for even scalar (`0e`) output
blocks. Pass `bias=True` to enable all eligible scalar biases or a boolean
sequence to select output blocks.

Compatible paths may be grouped into one MLX contraction for efficiency. This
is an implementation detail. The grouped execution path preserves derivatives
with respect to external weights, including repeated input and output irrep
blocks.

The upstream-compatibility suite checks instruction ordering, weight views,
shared and per-sample layouts, feature channels, normalization, compilation,
equivariance, and external-weight VJPs. The randomized parity harness also
compares forward values and VJPs against e3nn using identical generated arrays.
