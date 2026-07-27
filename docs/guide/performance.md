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
