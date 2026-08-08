# Performance and automatic differentiation

MLX work is lazy and asynchronous with respect to the CPU. Correct benchmarks
must materialize outputs before stopping the timer:

```python
start = time.perf_counter()
output = model(inputs)
mx.eval(output)
elapsed = time.perf_counter() - start
```

Run warm-up iterations before sampling so compilation and kernel caching are
not counted as steady-state inference. The repository's `evals/` harness does
this consistently across Torch CPU, general MLX, and kernel-enabled MLX.

## Generated Metal kernels

Tensor products, spherical harmonics, and scatter aggregation can use generated
Metal kernels. Dispatch is automatic: compatible kernels fuse indexing,
Clebsch–Gordan contraction, weighting, and accumulation; the general MLX path
handles unsupported inputs and performance crossover points.

Dense shared-weight tensor products deliberately keep their parameter vector
compact. Expanding a shared vector across the batch produces the same forward
values but creates a large per-item weight-gradient reduction. The compact
execution is used by FullyConnectedTensorProduct and by general shared-weight
TensorProduct instructions in every connection mode.

Set `use_custom_kernel=False` to compare paths or to request the most general
automatic-differentiation behavior. Disabling a kernel changes execution, not
the mathematical operation.

## Kernel limitations

Kernel-enabled execution is an adaptive policy, not a promise that every
operation uses a custom kernel. Unsupported shapes and measured crossover
points fall back to general MLX without changing results. Benchmark reports
record the selected path so a fallback is not reported as kernel execution.

- Generated kernels require Apple Silicon with Metal and currently operate on
  `float32`. Other devices and dtypes use general MLX.
- Tensor-product kernels require rank-two inputs with equal batch size.
  Weighted calls additionally require the exact supported shared rank-one or
  unshared per-item rank-two weight layout.
- Sparse scalar-path tensor products are capped at 2,000,000 generated terms,
  a batch size of 512, and 2,000,000 batch-times-path terms. Above these
  crossover guards, batched general MLX contractions are faster and are
  selected automatically. Dense shared-weight
  `FullyConnectedTensorProduct` therefore normally uses general MLX.
- The channel-local kernel is specialized for weighted `uvu` instructions
  with unique output blocks, uniform channel multiplicity, and unshared
  per-item weights. It is the important large-edge-count message-passing
  specialization; other connection modes use scalar-path kernels or general
  MLX.
- The generated spherical-harmonic kernel accepts nonempty rank-two
  `float32` vectors through `l=4`. Higher degrees, empty inputs, and additional
  leading dimensions use the general recurrence.
- Generated scatter accepts nonempty rank-two-or-higher `float32` sources and
  eager, valid indices. Its atomic accumulation can be slower than MLX
  indexed-add for some shapes, so it is opt-in and should be benchmarked for
  the target graph.
- Kernels currently operate on individual primitives. Message tensor
  products, radial weighting, and scatter are not fused into one graph-level
  dispatch, so intermediate message arrays still consume memory.

The shipped `gate_points_2102` model currently has no model-level kernel
toggle and always uses its general MLX tensor-product and scatter paths. The
v2106 models propagate the toggle, but remain mixed executions because dense
and unsupported operations fall back automatically.

Compilation and generated-kernel startup are reported separately by the
evaluation suite and excluded from steady-state latency. For short-lived or
small workloads, startup can outweigh the steady-state gain.

## Differentiation boundary

Reverse-mode gradients and reverse-over-reverse second derivatives are
supported by generated kernels. MLX cannot currently apply JVP directly to a
`CustomKernel` primitive. JVP users—Hessian-vector products, phonons,
directional response, tangent dynamics, and forward-mode sensitivity—should
use the general path:

```python
output = tensor_product.differentiable_arrays(left, right, weights)
harmonics = o3.spherical_harmonics(
    degrees, vectors, use_custom_kernel=False
)
```

For graph reduction, use
`scatter_sum(..., use_custom_kernel=False, jvp_safe=True)`. MLX 0.31's
indexed-add primitive does not implement JVP, so this fixed-index fallback uses
a sparse sorted prefix sum with linear memory. Ordinary scatter calls retain
the faster indexed-add implementation.

See the repository's
[evaluation guide](https://github.com/lamalab-org/e3nn_mlx/blob/main/evals/README.md)
for the compact three-way benchmark and randomized numerical parity suites.
