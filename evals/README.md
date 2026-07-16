# e3nn-mlx versus PyTorch/e3nn performance evaluations

## What this suite answers

This directory measures equivalent equivariant work in the MLX port and in
upstream PyTorch/e3nn on Apple Silicon. It is designed to answer three separate
questions:

1. Which low-level operations currently favor MLX, PyTorch MPS, or PyTorch CPU?
2. Does MLX graph compilation amortize its cold-start cost for realistic point
   models?
3. At what tensor, edge, and model sizes do launch overhead, memory bandwidth,
   or tensor-product arithmetic become the bottleneck?

It does **not** start with a promised speedup. Every gain in the generated
report is computed as:

```text
speedup = synchronized PyTorch eager median / synchronized MLX median
```

A result above `1.0x` favors MLX; below `1.0x` favors PyTorch. Raw samples,
quartiles, cold compilation time, workload dimensions, framework versions, and
non-identifying hardware information are retained in JSON.

## Compared operations

| Case | Equivalent work | Why it matters |
| --- | --- | --- |
| `spherical_harmonics` | All degrees from `0` through `lmax`, component normalization | Executed for every geometric edge; many small polynomial kernels can be launch-bound. |
| `full_tensor_product` | Unweighted `FullTensorProduct` of the same input irreps | Materializes all representation-product channels and strongly stresses Clebsch--Gordan contractions. |
| `fully_connected_tensor_product` | Learned `FullyConnectedTensorProduct` with identical input/output irreps | Models the dense equivariant mixing used by learned interactions. |
| `linear` | Blockwise equivariant `Linear` with the same irreps | Isolates multiplicity mixing from geometry and sparse aggregation. |
| `v2106_convolution` | Radial MLP, tensor product, scatter, self connection, and learned alpha on a fixed graph | The core sparse interaction in the modular v2106 models. |
| `v2106_message_passing` | A gated stack of v2106 convolutions | Exposes fusion and launch overhead across repeated interactions. |
| `v2106_network` | Position geometry, radial basis, spherical harmonics, attributed message passing, and graph pooling | The closest end-to-end comparison of actual model execution. |

Forward mode uses inference semantics on PyTorch. Training mode computes
gradients of all trainable parameters for learned modules; parameter-free
spherical harmonics and full tensor products compute input gradients. Neither
mode updates weights, so optimizer choice is not mixed into the operator
comparison.

The sparse model cases use the exact same deterministic directed ring topology
and equivalent representation shapes. Topology construction is deliberately
outside the timed region: the port's `radius_graph` and the upstream reference
are both simple quadratic helpers, whereas production molecular models usually
receive a neighbor list from a specialized data pipeline. Edge geometry,
spherical harmonics, radial features, scatter, gates, and pooling remain timed.

## Environment setup

The two frameworks run in separate processes. This prevents one framework's
allocator and compiled graphs from contaminating the other's unified-memory
measurements, and permits PyTorch to use Python 3.12 while the project MLX
environment uses Python 3.14.

The existing project environment is used for MLX:

```bash
/Users/rostislav/Documents/Lamalab/Homebrew/e3nn_mlx/.venv/bin/python \
  evals/run_backend.py --backend mlx --preset smoke \
  --output evals/results/manual-mlx.json
```

Create the isolated, pinned reference environment with `uv`:

```bash
uv venv evals/.venv-torch --python 3.12
uv pip install --python evals/.venv-torch/bin/python \
  -r evals/requirements-torch.txt
```

The pinned versions match the repository's numerical reference environment:
PyTorch 2.7.1 and e3nn 0.5.8. The JSON metadata records the versions actually
used, so experimenting with a newer PyTorch wheel does not silently invalidate
comparisons.

## Quick smoke comparison

Run all small cases on MLX GPU and PyTorch MPS, then generate plots:

```bash
python3 evals/run.py --backend both --preset smoke --plot
```

Add the CPU reference to the same isolated comparison with:

```bash
python3 evals/run.py --backend both --torch-device both --preset smoke --plot
```

This launches three workers: MLX, `torch-mps`, and `torch-cpu`. The Torch
workers are separate processes so CPU allocator state and MPS unified-memory
state do not contaminate one another. Their JSON rows and plot labels remain
distinct. To compare only MLX with Torch CPU, use `--torch-device cpu`; to run
only the CPU reference, use `--backend torch --torch-device cpu`.

The default interpreters are:

- MLX: `.venv/bin/python`
- PyTorch: `evals/.venv-torch/bin/python`

Override them when needed:

```bash
python3 evals/run.py --backend both --preset smoke \
  --mlx-python /path/to/mlx-python \
  --torch-python /path/to/torch-python \
  --torch-device both --plot
```

PyTorch eager is the default reference because `torch.compile` support and
benefit on MPS can vary by operation. Add `--torch-compile` to record separate
compiled PyTorch rows; failures are preserved in the JSON rather than being
mistaken for a slow result.

For a fast development check, select cases and shorten sampling:

```bash
python3 evals/run.py --backend mlx --preset smoke \
  --case spherical_harmonics --case v2106_convolution \
  --warmup 1 --samples 2 --plot
```

## Presets and large-scale experiment

`smoke` proves that every path constructs, compiles, differentiates, times, and
serializes. Its numbers are too noisy and too launch-dominated for performance
claims. `medium` is the recommended first comparison. `large` is intended for
a plugged-in Mac with tens of gigabytes of unified memory and should be run one
case at a time.

| Preset | Representative scale | Samples | Intended use |
| --- | --- | --- | --- |
| `smoke` | 48--512 nodes/items, small multiplicities | 7 | Functional validation only |
| `medium` | Up to 65k directions, 2k nodes, 32k edges | 15 | Primary latency/throughput comparison |
| `large` | Up to 1M directions, 8k nodes, 196k edges | 20 | Saturation, memory pressure, and scaling |

Recommended large experiment:

```bash
# Reboot or close GPU-heavy applications, connect power, and run each case
# separately so allocator peaks and thermal behavior are attributable.
for case in spherical_harmonics linear full_tensor_product \
            fully_connected_tensor_product v2106_convolution \
            v2106_message_passing v2106_network; do
  python3 evals/run.py --backend both --preset large \
    --case "$case" --warmup 8 --samples 30
done

# Plot any collection of JSON artifacts together.
python3 evals/plot_results.py evals/results/*/combined.json \
  --output-dir evals/results/large-comparison-plots
```

For a three-way large-scale comparison, add `--torch-device both` to each
invocation. CPU runs can take much longer than GPU runs at this scale; first
validate the command with `smoke`, then use `medium`, and reserve `large` for
the cases where the CPU baseline is scientifically useful. Keep CPU frequency,
power mode, and background load stable, and alternate `--backend-order` across
repetitions to reduce thermal-order bias.

Start with forward-only runs if memory is uncertain:

```bash
python3 evals/run.py --backend both --preset large \
  --case v2106_network --phase forward --plot
```

Any preset value can be overridden without editing source:

```bash
python3 evals/run.py --backend both --preset medium \
  --case v2106_convolution \
  --set v2106_convolution.nodes=4096 \
  --set v2106_convolution.neighbors=24 \
  --set v2106_convolution.mul=32 --plot
```

For a scaling study, repeat that command over powers of two for `items` or
`nodes`, and over `lmax = 1, 2, 3, 4`. Keep every other parameter fixed. This
distinguishes constant launch overhead from asymptotic tensor-product cost.

## Timing methodology

- Every GPU invocation is synchronized with `mx.synchronize()` or
  `torch.mps.synchronize()` before the wall-clock sample ends. Torch CPU calls
  are synchronous, so returning from the call is their timing boundary.
- Warmups happen before recorded samples.
- The report uses the median; raw samples and interquartile ranges remain in
  JSON for variance inspection.
- MLX eager and MLX compiled are separate rows. The first compiled invocation
  is timed separately as `compile_ms` and excluded from steady-state latency.
- PyTorch forward uses inference mode. Training includes forward, scalar loss,
  reverse mode, and parameter-gradient materialization.
- Inputs and topology are allocated before timing.
- MLX peak memory is reset and recorded per mode. PyTorch MPS does not expose an
  equivalent resettable peak metric in the pinned interface, so its field is
  left empty instead of presenting incomparable allocator numbers.
- The suite records hardware model, chip, core description, and physical
  memory, but intentionally excludes serial numbers and machine identifiers.

For publication-quality numbers, use at least 30 samples, report medians and
quartiles, run three fresh processes per configuration, randomize backend run
order, and monitor thermals. A single sequential run can be biased by shader
cache state and thermal throttling. Alternate `--backend-order mlx-first` and
`--backend-order torch-first` between repetitions.

## Hypothetical gains worth testing

These are hypotheses for experiment design, not measured promises:

- **Small fused pipelines:** MLX compilation may deliver roughly `1.5--4x`
  over PyTorch eager when many tiny spherical-harmonic, gate, scatter, and
  elementwise kernels are launch-bound. The advantage should grow with layer
  count until arithmetic dominates.
- **Medium end-to-end inference:** a plausible target is `1.2--3x` for fixed
  topology if MLX fuses surrounding operations and avoids framework overhead.
- **Training:** `1.1--2.5x` is plausible for repeatedly reused compiled shapes,
  but gradient tensor products and scatter may narrow or reverse the gain.
- **Large dense tensor products:** PyTorch/e3nn may match or beat this port.
  Upstream e3nn has mature generated contraction paths, while the current MLX
  implementation still expresses paths through general array operations. A
  `0.3--1.2x` MLX/PyTorch range would not be surprising here.
- **Very small or one-shot calls:** cold compilation can make MLX slower even
  when its steady-state kernel is faster. Break-even iterations are roughly
  `compile_ms / (torch_ms - mlx_ms)` when the denominator is positive.
- **Memory-bound large workloads:** gains should contract toward hardware
  bandwidth limits. Unified memory removes explicit host/device copies for both
  MPS and MLX, so it is not by itself evidence of an MLX speedup.

The most valuable outcome may be a mixed profile rather than one headline
number: for example, fast compiled spherical harmonics and gating but a tensor
product bottleneck. That directly identifies which MLX kernels deserve custom
optimization.

## Outputs and plots

Each run creates a timestamped directory under `evals/results/` containing
backend JSON files and `combined.json`. `--plot` adds:

- `latency_forward.svg` and `latency_train.svg`;
- `speedup_forward.svg` and `speedup_train.svg`, with separate Torch MPS and
  Torch CPU reference bars when both are present;
- `compile_cost.svg`;
- `peak_memory.svg`;
- `scaling_<case>.svg` log-log throughput curves when multiple sizes of a case
  are supplied;
- `summary.csv`;
- `report.html` embedding all plots and run metadata.

Plotting uses only the Python standard library and produces portable SVG, so no
Matplotlib environment is required:

```bash
python3 evals/plot_results.py run-a/combined.json run-b/combined.json \
  --output-dir comparison-plots
```

Result artifacts and the reference virtual environment are git-ignored. Keep
the JSON alongside any reported chart: the chart alone does not preserve
versions, workload shapes, sample variance, or compilation cost.

## Background references

- [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
