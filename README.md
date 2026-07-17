# e3nn-mlx

Incremental refactor of e3nn into a backend-neutral core plus an MLX-native runtime.

The P0 implementation currently provides:

- backend-agnostic O(3) metadata in `e3nn_core`
- e3nn-compatible real Wigner-3j and Clebsch--Gordan coefficients
- batched real Wigner matrices and O(3) parity transforms through arbitrary `l`
- spherical harmonics generated recursively from the canonical Wigner basis
- weighted and unweighted MLX tensor products and standard wrappers
- lightweight MLX Linear, Gate, Norm, and reduction operations
- graph/radial utilities, the gated `gate_points_2102` network, and the complete
  modular `v2106` point-model family
- pinned upstream e3nn numerical fixtures and Apple-Silicon CI coverage

P0 guarantees numerical compatibility through `l=6`; reference and structural
tests exercise selected operations through `l=8`.  See
[`docs/P0_COMPATIBILITY.md`](docs/P0_COMPATIBILITY.md) for conventions, shapes,
normalization, and the release gate.

## e3nn-style API

New code can use the familiar upstream namespace layout with ordinary MLX
arrays:

```python
import mlx.core as mx
from e3nn_mlx import nn, o3

linear = o3.Linear("16x0e + 16x1o", "32x0e + 16x1o")
x = mx.random.normal((128, linear.irreps_in.dim))
y = linear(x)
```

`e3nn_mlx.o3`, `e3nn_mlx.nn`, `e3nn_mlx.math`, and
`e3nn_mlx.nn.models.v2106` mirror the implemented PyTorch/e3nn organization.
Raw inputs return raw MLX arrays; `IrrepsArray` inputs preserve typed outputs.
The original flat API remains backward-compatible. See
[`docs/HIGH_LEVEL_API.md`](docs/HIGH_LEVEL_API.md) for migration examples,
model imports, compilation, deliberate differences, and no-overhead evidence.

## Development

Install the MLX and test extras and run the suite on Apple Silicon:

```bash
python -m pip install -e '.[mlx,test]'
python -m pytest
```

MLX tests may skip when Metal is unavailable locally.  The required macOS CI job
sets `E3NN_MLX_REQUIRE_RUNTIME=1`, which turns such skips into failures.

## Gate-points model

`e3nn_mlx.models.gate_points_2102.Network` ports the upstream gated point-cloud
network. It accepts a dictionary containing `pos`, `x`, `z`, and an optional
integer `batch` array and returns an `IrrepsArray`. `GatePointsNetwork` and
`GatePointsConvolution` are also exported from `e3nn_mlx`.

Neighbor discovery with `radius_graph` is eager because MLX 0.31 does not have
a device-side dynamic nonzero operation. For compiled training or inference,
construct the topology once and call `Network.forward_with_edges`; the radial
embedding, spherical harmonics, message passing, gating, and graph reduction
then remain inside the compiled MLX graph.

The complete modular June 2021 family is available from
`e3nn_mlx.models.v2106`. It includes `Convolution`, `MessagePassing`,
`SimpleNetwork`, and `NetworkForAGraphWithAttributes`; both networks support
eager dictionary input and compiled fixed-edge execution. See
[`docs/V2106_POINT_MODELS.md`](docs/V2106_POINT_MODELS.md) for their APIs,
compilation boundary, upstream semantics, and verification coverage.

## Performance evaluation

The isolated cross-framework harness in [`evals/`](evals/README.md) compares
MLX eager/compiled execution with upstream PyTorch/e3nn on Apple Silicon. It
covers spherical harmonics, tensor products, Linear, v2106 convolution,
message passing, and an end-to-end attributed network in forward and training
modes. Small smoke presets, larger scaling experiments, synchronized timing,
JSON/CSV results, and dependency-free SVG/HTML plot generation are included.
Generated Metal kernels and the reproducible kernel/general-MLX/Torch-CPU
comparison are documented in
[`evals/KERNEL_EVALUATION.md`](evals/KERNEL_EVALUATION.md).

### JVP and generated-kernel boundary

A Jacobian-vector product (JVP) propagates a chosen input perturbation through
a function without constructing its complete Jacobian. Most users do not call
JVP directly: ordinary inference and standard training—including MACE energy,
force, and parameter-gradient training—use forward evaluation and reverse-mode
gradients.

JVP is useful for more specialized atomistic workflows, including:

- Hessian-vector products and directional force-constant calculations;
- phonon, vibrational-response, and stability algorithms that propagate a
  displacement direction;
- mixed position/parameter response calculations;
- tangent dynamics, sensitivity analysis, and forward-mode Jacobian APIs;
- debugging equivariance by differentiating along an infinitesimal rotation.

MLX 0.31 cannot currently apply JVP directly to a `CustomKernel` primitive.
For these workflows, select the fully differentiable MLX implementation:

```python
import e3nn_mlx

# Spherical harmonics and graph reduction
y = e3nn_mlx.spherical_harmonics(
    degrees, vectors, use_custom_kernel=False
)
summed = e3nn_mlx.scatter_sum(
    messages, edge_dst, num_nodes, use_custom_kernel=False
)

# TensorProduct: use this callable inside mx.jvp
y = tensor_product.differentiable_arrays(left, right, weights)
```

This changes execution strategy, not mathematical conventions or accuracy.
Reverse-mode gradients and reverse-over-reverse second derivatives remain
supported by the generated kernels. See
[`evals/KERNEL_EVALUATION.md`](evals/KERNEL_EVALUATION.md) for the performance
and fallback details.
