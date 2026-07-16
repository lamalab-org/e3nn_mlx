# Weighted tensor-product optimization report

This report records the July 16, 2026 optimization audit for the five proposed
MLX improvements. The measured system was an Apple M4 Pro MacBook Pro with
48 GB unified memory, MLX 0.31.1, Python 3.14.6, and macOS 26.5.1.

## Implemented work

1. **Fuse weighted tensor-product contractions.** Compatible `uvu` and `uvv`
   paths now form the input outer product once, apply a combined
   Clebsch--Gordan basis transform, and immediately reduce it with the learned
   path weights.
2. **Group compatible weighted instructions.** Instructions sharing input
   blocks and connection mode reuse one coupling transform. Different output
   irreps are sliced from the combined result without recomputing the input
   coupling.
3. **Evaluate contraction-plus-aggregation.** The retained implementation uses
   MLX's native indexed add. A custom atomic Metal scatter was measured and
   rejected: medium compiled scatter forward regressed from about 2.25 ms to
   7.81 ms, and convolution forward regressed from about 33.2 ms to 38.5 ms.
   Avoiding the logical edge-message array will require a genuinely fused
   tensor-product/scatter kernel with a custom backward pass; replacing only
   scatter is counterproductive. The new `scatter_sum` case keeps this boundary
   measurable.
4. **Specialize equivariant `Linear`.** Paths for each identical irrep are
   assembled into a block matrix and executed with one matrix multiplication,
   instead of one general contraction per instruction.
5. **Add scalar-irrep shortcuts.** Scalar-to-irrep and irrep-to-scalar coupling
   uses diagonal broadcast multiplication. Gate applies activated scalar gates
   directly to irrep chunks instead of constructing a temporary elementwise
   tensor product.

The general instruction-wise tensor-product implementation remains the
fallback. In particular, `uvw` is deliberately excluded from the grouped path:
grouping it increased intermediate broadcast size and made both execution time
and memory use worse.

## Medium-preset results

MLX figures below are compiled steady-state medians from 15 samples after five
warmups. Each final workload ran in a fresh process. “Old MLX gain” compares
the same compiled workload before these changes; “Torch CPU gain” divides the
upstream e3nn/PyTorch eager CPU median by the optimized MLX compiled median.

| Workload | MLX forward | MLX train | Old MLX gain, forward / train | Torch CPU gain, forward / train |
| --- | ---: | ---: | ---: | ---: |
| weighted `uvu` tensor product | 26.98 ms | 129.57 ms | 4.23× / 2.58× | 4.55× / 3.86× |
| scatter sum | 2.26 ms | 5.12 ms | 0.99× / 1.03× | 8.93× / 4.63× |
| gate | 2.60 ms | 14.03 ms | 10.36× / 5.69× | 3.61× / 2.82× |
| radial MLP | 0.87 ms | 3.94 ms | 0.99× / 1.04× | 2.23× / 2.65× |
| equivariant Linear | 2.45 ms | 4.89 ms | 1.26× / 1.35× | 2.58× / 2.84× |
| v2106 convolution | 32.19 ms | 111.23 ms | 3.65× / 2.76× | 4.85× / 3.49× |
| v2106 message passing | 54.29 ms | 187.00 ms | 3.70× / 2.81× | 5.27× / 4.17× |
| v2106 network | 126.90 ms | 624.91 ms | 3.98× / 2.79× | 4.54× / 3.30× |

The unchanged scatter and radial-MLP rows are controls. They show that the
end-to-end gain comes from equivariant contraction, instruction reuse, Linear,
and Gate rather than a machine-wide timing shift. The scatter result also
supports retaining native MLX indexed add.

These are strong development results, not publication-quality confidence
intervals. A formal comparison should repeat each configuration in at least
three fresh processes, alternate backend order, and report quartiles from the
generated JSON.

## Reproduction

Run an isolated optimized MLX case with:

```bash
.venv/bin/python evals/run_backend.py \
  --backend mlx --preset medium \
  --case weighted_tensor_product_uvu \
  --warmup 5 --samples 15 \
  --output /tmp/weighted-uvu.json
```

Run the three-way MLX, Torch MPS, and Torch CPU comparison with plots using:

```bash
python3 evals/run.py \
  --backend both --torch-device both \
  --preset medium --warmup 5 --samples 15 --plot
```

For lower-noise model measurements, select one case per invocation. The main
evaluation README documents scaling overrides and the large preset.
