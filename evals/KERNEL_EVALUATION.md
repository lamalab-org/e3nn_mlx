# Generated Metal kernel evaluation

This branch adds generated, shape-specialized Metal operations while preserving
the existing array implementation behind `use_custom_kernel=False`. The two
paths are deliberately benchmarked in isolated processes: this prevents shader
caches, compiled MLX graphs, and allocator state from making the kernel baseline
look better than it is.

## What is implemented

| Area | Generated operation | Selection policy |
| --- | --- | --- |
| Tensor-product instructions | Sparse nonzero CG paths fuse input indexing, path weights, contraction, and output accumulation | Used for supported float32 rank-2 products up to the large dense-path guard |
| Weighted `uvu` tensor product | One thread handles one item/channel and executes all compatible instructions | Used at large edge counts; this is the principal model-oriented kernel |
| `FullTensorProduct` | Static sparse scalar paths with grouped output mapping | Used through 512 items; larger dense batches retain the faster batched-MLX implementation |
| `FullyConnectedTensorProduct` | The same generated weighted path engine | Very dense/high-multiplicity plans retain MLX matrix contractions when expanded metadata would exceed two million terms |
| Spherical harmonics | Coordinate normalization and recurrence polynomial evaluation are fused per vector; the backward kernel evaluates the analytic Jacobian | Float32 rank-2 inputs through `l=4`; other shapes/degrees use the general recurrence |
| Scatter sum | Atomic generated scatter plus a generated gather transpose | Experimental opt-in only: MLX indexed add is faster on the tested workloads |

Generated tensor-product kernels store only nonzero CG coefficients and their
integer indices. Kernel objects and contraction metadata are cached by static
signature. Unsupported dtype, rank, broadcasting, weight layout, degree, or
path density automatically uses the general implementation.

The generated reverse-mode rules support ordinary training and second reverse
derivatives. MLX 0.31 cannot currently apply JVP directly to a `CustomKernel`
primitive even when its containing custom function supplies a JVP rule. Code
which needs forward-mode transformation should therefore call tensor-product
`differentiable_arrays`, or pass `use_custom_kernel=False` to spherical
harmonics. For fixed-topology scatter, pass `use_custom_kernel=False` and
`jvp_safe=True`; this avoids MLX's missing indexed-add JVP with a sparse sorted
prefix sum. This is an explicit execution boundary, not a numerical
approximation.

Typical JVP users are phonon and vibrational-response calculations,
Hessian-vector products, mixed position/parameter response, tangent dynamics,
and developers differentiating along infinitesimal rotations to diagnose
equivariance. Standard inference and reverse-mode energy/force training do not
require a direct JVP. The root README contains concrete fallback examples.

## Three-way comparison

Use `--mlx-kernels both` to launch independent generated-kernel and general-MLX
workers. `--torch-device cpu` adds upstream e3nn/PyTorch CPU in a third process:

```bash
python3 evals/run.py \
  --backend both --torch-device cpu --mlx-kernels both \
  --preset medium \
  --case spherical_harmonics \
  --case full_tensor_product \
  --case fully_connected_tensor_product \
  --case weighted_tensor_product_uvu \
  --case scatter_sum \
  --warmup 8 --samples 30 --plot
```

The result uses the unambiguous backend labels `mlx-kernel`,
`mlx-no-kernel`, and `torch-cpu`. Both MLX workers report eager and compiled
rows. Torch CPU remains eager unless `--torch-compile` is explicitly supplied.

For a focused scaling study of the most successful generated contraction:

```bash
for nodes in 512 1024 2048 4096 8192; do
  python3 evals/run.py \
    --backend both --torch-device cpu --mlx-kernels both \
    --preset medium --case weighted_tensor_product_uvu \
    --set weighted_tensor_product_uvu.nodes=$nodes \
    --warmup 8 --samples 30
done

python3 evals/plot_results.py evals/results/*/combined.json \
  --output-dir evals/results/kernel-scaling-plots
```

## Large-scale experiment

Run one expensive case per fresh process on a plugged-in, thermally stable Mac:

```bash
for case in spherical_harmonics full_tensor_product \
            fully_connected_tensor_product weighted_tensor_product_uvu \
            scatter_sum v2106_convolution; do
  python3 evals/run.py \
    --backend both --torch-device cpu --mlx-kernels both \
    --preset large --case "$case" \
    --warmup 12 --samples 50 --plot
done
```

First run the identical command with `--preset smoke --warmup 2 --samples 3`.
Smoke validates construction, compilation, forward, backward, serialization,
and plotting; it is too launch-bound for performance conclusions.

## Current measured direction

An exploratory medium run on the development Mac measured the compiled weighted
`uvu` contraction at 6.96 ms forward and 49.42 ms training, versus 27.15 ms and
135.06 ms for the isolated no-kernel MLX path. That is a 3.90x forward gain and
a 2.73x training gain. The prior same-workload Torch CPU reference was 124.01 ms
and 507.25 ms, corresponding to approximately 17.8x and 10.3x respectively.
These are development measurements, not release claims; rerun the 30-sample
command above and retain its JSON before publishing them.

The experiments also rejected two tempting defaults:

- the scalar `FullTensorProduct` kernel loses to batched MLX contractions at
  large item counts, so the runtime switches back above the measured crossover;
- atomic scatter was about 1.8x slower than MLX indexed add in the medium test,
  so it remains opt-in and is not used by v2106 convolution.

This selection discipline matters: a custom kernel is useful only if the full
synchronized operation improves, not merely because it combines more source
expressions into one shader.
