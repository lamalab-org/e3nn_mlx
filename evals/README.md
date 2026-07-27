# Evaluation suite

This directory has two jobs:

1. compare the performance of upstream e3nn on Torch CPU, general compiled
   MLX, and compiled MLX with generated kernels;
2. qualify numerical compatibility with seeded randomized parity tests.

The performance runner always launches exactly these isolated workers:

| Result label | Implementation |
| --- | --- |
| `torch-cpu` | `torch==2.7.1`, `e3nn==0.5.8`, eager CPU execution using all available CPU threads |
| `mlx` | compiled e3nn-mlx with generated kernels disabled |
| `mlx-kernel` | compiled e3nn-mlx with generated kernels enabled |

There is no MPS mode, Torch compilation mode, MACE suite, or model-specific
benchmark in this directory.

## Setup

Use the project environment for both the orchestrator and MLX workers:

```bash
python -m pip install -e '.[test]'
```

Create the isolated Torch reference environment once:

```bash
uv venv evals/.venv-torch --python 3.12
uv pip install --python evals/.venv-torch/bin/python \
  -r evals/requirements-torch.txt
```

## Smoke test

The smoke preset proves that every case constructs, differentiates, compiles
where appropriate, runs in all three workers, writes JSON, and generates plots:

```bash
.venv/bin/python evals/run.py --preset smoke
```

Smoke timings are intentionally short and are not suitable for performance
claims.

## Full benchmark

Run the complete performance comparison with eight warmups and thirty recorded
samples:

```bash
.venv/bin/python evals/run.py --preset full
```

The full preset measures both forward and training latency. To reduce memory or
runtime while investigating one operation, select cases or phases:

```bash
.venv/bin/python evals/run.py --preset full \
  --case weighted_tensor_product_uvu \
  --phase forward
```

Settings can be overridden without editing source:

```bash
.venv/bin/python evals/run.py --preset full \
  --case spherical_harmonics \
  --set spherical_harmonics.items=524288
```

## Operations

| Case | Purpose |
| --- | --- |
| `spherical_harmonics` | Geometry encoding used for every edge |
| `full_tensor_product` | All unweighted Clebsch–Gordan output paths |
| `fully_connected_tensor_product` | Dense learned equivariant mixing |
| `weighted_tensor_product_uvu` | Per-item learned contraction used by message passing |
| `linear` | Equivariant multiplicity mixing and non-kernel control |
| `scatter_sum` | Graph message aggregation |

The `linear` result is deliberately a control: it has no generated kernel, so
the two MLX workers should agree up to ordinary timing noise.

## Measurement and output

Torch CPU uses `os.cpu_count()` threads. MLX uses compiled execution for both
workers, while Torch uses upstream eager execution. Warmups occur before
recording. Every MLX sample is synchronized before its wall-clock interval
ends. Reports use the median and retain raw samples and quartiles.

Each run creates:

```text
evals/results/<timestamp>/
├── torch-cpu.json
├── mlx.json
├── mlx-kernel.json
├── combined.json
└── plots/
    ├── latency_forward.svg
    ├── latency_train.svg
    ├── speedup_forward.svg
    ├── speedup_train.svg
    ├── summary.csv
    └── report.html
```

The speedup is:

```text
Torch CPU median / MLX median
```

Values above `1×` favor MLX. Compilation time is reported separately and is
excluded from steady-state latency.

## Randomized numerical parity

[RANDOMIZED_PARITY.md](RANDOMIZED_PARITY.md) documents the deep randomized
TensorProduct harness and the complementary core-operation harness.

[RANDOMIZED_OPERATION_PARITY.md](RANDOMIZED_OPERATION_PARITY.md) documents
forward-and-VJP comparison across the broader numerical API, including
rotations, spherical harmonics, neural layers, reduced products, scatter, and
tensor-product wrappers.
