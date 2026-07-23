# e3nn-style high-level API

## Goal

The public compatibility layer follows the organization and ordinary-array
calling style used by PyTorch/e3nn while retaining MLX-native execution:

```python
import mlx.core as mx
from e3nn_mlx import nn, o3
from e3nn_mlx.math import soft_one_hot_linspace

irreps = o3.Irreps("16x0e + 16x1o")
linear = o3.Linear(irreps, "32x0e + 16x1o")
x = mx.random.normal((128, irreps.dim))
y = linear(x)  # raw MLX array, as in torch/e3nn
```

The compatibility layer delegates directly to the existing, tested numerical
kernels. It does not reimplement rotations, contractions, scatter operations,
or model layers.

## Namespace mapping

| PyTorch/e3nn | e3nn-mlx |
| --- | --- |
| `from e3nn import o3` | `from e3nn_mlx import o3` |
| `from e3nn import nn` | `from e3nn_mlx import nn` |
| `from e3nn.math import ...` | `from e3nn_mlx.math import ...` |
| `e3nn.nn.models.v2106` | `e3nn_mlx.nn.models.v2106` |
| `o3.wigner_D(...)` | `o3.wigner_D(...)` |
| `module.forward(x)` | `module.forward(x)` |

The implemented `o3` namespace includes metadata, rotations, spherical
harmonics, Linear, Norm, tensor-product families, reduced tensor products,
S2 grids, and SO3 grids. The `nn` namespace includes the implemented
activations, Gate, BatchNorm, Dropout, extraction, normalized activation, and
fully connected network helpers.

Only functionality already implemented by e3nn-mlx is exported. The namespace
layout is familiar, but it does not claim that every upstream e3nn class is
present.

## Raw and typed array behavior

High-level modules preserve the caller's style:

```python
raw_output = linear(x)
assert isinstance(raw_output, mx.array)

typed_x = o3.IrrepsArray(irreps, x)
typed_output = linear(typed_x)
assert isinstance(typed_output, o3.IrrepsArray)
assert typed_output.irreps == linear.irreps_out
```

This makes ordinary model code resemble PyTorch/e3nn while keeping
`IrrepsArray` available at representation-sensitive boundaries. Mixed raw and
typed inputs to a binary tensor product return a typed result; a raw result is
returned only when all representation-carrying inputs are raw.

The original flat API remains unchanged:

```python
import e3nn_mlx as e3nn

# Existing code continues to use the representation-aware class.
legacy_linear = e3nn.Linear("1o", "1o")
typed_output = legacy_linear(e3nn.IrrepsArray("1o", vectors))
```

This separation avoids silently changing return types in existing projects.

## Tensor products

The upstream-shaped call uses raw arrays:

```python
tp = o3.FullyConnectedTensorProduct("8x1o", "8x1o", "8x0e + 8x1e + 8x2e")
output = tp(left, right)
```

Shared/unshared weights, instruction formats, normalization, `output_mask`,
`weight_numel`, `weight_views`, and `right` retain the existing implementation
semantics. `TensorSquare` similarly accepts a single raw array.

## Neural-network modules

```python
gate = nn.Gate(
    "16x0e",
    [mx.tanh],
    "16x0e",
    [mx.sigmoid],
    "16x1o",
)
output = gate(features)
```

`Activation`, `BatchNorm`, `Dropout`, `Gate`, `Identity`, `NormActivation`,
`Extract`, and `ExtractIr` automatically attach their configured input irreps
for raw arrays. Their explicit `forward` methods and callable behavior are
equivalent.

## Model imports

The upstream module layout is mirrored:

```python
from e3nn_mlx.nn.models.v2106 import SimpleNetwork
from e3nn_mlx.nn.models.v2106.points_convolution import Convolution
from e3nn_mlx.nn.models.v2106.gate_points_message_passing import MessagePassing
```

Models imported from `e3nn_mlx.nn.models` return raw MLX arrays when their node
features and attributes are raw. The original `e3nn_mlx.models` imports retain
their `IrrepsArray` results.

```python
network = SimpleNetwork(
    "3x0e + 2x1o",
    "4x0e + 1x1o",
    max_radius=2.0,
    num_neighbors=3.0,
    num_nodes=5.0,
)
output = network({"pos": positions, "x": features})
```

Fixed-topology compilation remains available through the original
`forward_with_edges` method. Its representation-aware result is intentional;
the upstream-style dictionary `forward`/call boundary performs the raw-output
conversion.

## Compilation and performance

The raw/typed decision is a Python type check performed while constructing or
tracing the MLX graph. The compiled graph contains the same numerical kernels
as the original API:

```python
compiled_linear = mx.compile(linear)
compiled_output = compiled_linear(x)
```

No array copy is introduced. `IrrepsArray` stores the original MLX array and
static metadata, and unwrapping returns that same array object. On the local M4
Pro regression benchmark (`8192 x (16x0e + 16x1o + 16x2e)` Linear, 40 synchronized
samples), the adapter and original paths were within measurement noise:

| Path | Median |
| --- | ---: |
| Original eager | 0.466 ms |
| High-level eager | 0.438 ms |
| Original compiled | 0.393 ms |
| High-level compiled | 0.373 ms |

These figures are a no-regression check, not a general performance claim. The
cross-framework experiments in the repository's
[`evals/` guide](https://github.com/lamalab-org/e3nn_mlx/blob/main/evals/README.md)
remain the proper tool for model-scale measurements.

## Deliberate differences from PyTorch/e3nn

- Arrays, modules, gradients, serialization, and compilation use MLX.
- The package remains named `e3nn_mlx`; it does not shadow or replace an
  installed `e3nn` package.
- `Irreps.randn` is not added to the backend-neutral metadata class. Allocate
  with `mx.random.normal((..., irreps.dim))` instead.
- Eager radius-graph discovery remains outside compiled execution because MLX
  0.31 has no device-side dynamic `nonzero` operation.
- The compatibility namespaces expose the implemented surface, not unported
  upstream features.

## Verification

`tests/test_high_level_api.py` verifies namespace imports, raw/typed output
preservation, exact equality with the original implementation, tensor-product
families, neural-network modules, gradients, compilation, explicit `forward`,
model module paths, and unchanged legacy behavior. The complete numerical and
equivariance suite remains the release gate.
