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
this consistently across MLX, PyTorch CPU, and PyTorch MPS.

## Generated Metal kernels

Tensor products, spherical harmonics, and scatter aggregation can use generated
Metal kernels. Dispatch is automatic: compatible kernels fuse indexing,
Clebsch–Gordan contraction, weighting, and accumulation; the general MLX path
handles unsupported inputs and performance crossover points.

Set `use_custom_kernel=False` to compare paths or to request the most general
automatic-differentiation behavior. Disabling a kernel changes execution, not
the mathematical operation.

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

Scatter is a separate boundary: MLX 0.31's indexed-add primitive does not
implement JVP, so `scatter_sum` supports reverse-mode differentiation but has
no forward-mode fallback. Compute a message JVP before aggregation or use a
problem-specific fixed incidence matrix when the reduction itself needs JVP.

See the repository's
[kernel evaluation guide](https://github.com/lamalab-org/e3nn_mlx/blob/main/evals/KERNEL_EVALUATION.md)
for reproducible comparisons and the exact fallback contract.
