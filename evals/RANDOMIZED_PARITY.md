# Randomized TensorProduct parity qualification

`randomized_tensor_product_parity.py` generates seeded NumPy values and feeds
the identical arrays to pinned Torch/e3nn and e3nn-mlx workers. The workers run
in isolated environments so Torch is not added to the MLX package or normal
test dependencies.

This complements the committed golden fixtures. Golden fixtures provide a
stable release contract; this harness explores a much larger structural space.

## What is randomized

Each valid case varies:

- all eight connection modes: `uvw`, `uvu`, `uvv`, `uuw`, `uuu`, `uvuv`,
  `uvu<v`, and `u<vw`;
- angular momenta, parity, multiplicities, repeated irreps, unrelated blocks,
  and zero-multiplicity blocks;
- one to four instructions, including repeated output targets and mixed
  weighted/unweighted paths where upstream supports them;
- component/norm irrep normalization, element/path normalization, and positive
  user path weights;
- vector, ordinary batch, singleton-weight, and multidimensional broadcast
  layouts with shared or per-sample weights;
- MLX generic eager, generic compiled, and custom-kernel-requested execution.

The generator derives mode-specific multiplicities rather than discarding
invalid random combinations after construction. A fixed seed reproduces the
complete structures and numerical values.

## Setup

Create the pinned Torch reference environment if it does not already exist:

```bash
uv venv evals/.venv-torch --python 3.12
uv pip install --python evals/.venv-torch/bin/python \
  -r evals/requirements-torch.txt
```

The default MLX interpreter is the interpreter used to launch the parent
script. The default Torch interpreter is `evals/.venv-torch/bin/python` when
that file exists.

## Commands

Run a small qualification:

```bash
.venv/bin/python evals/randomized_tensor_product_parity.py \
  --cases 64 --seed 20260727 --max-l 3 --max-mul 3
```

Run a broader qualification and specify both environments explicitly:

```bash
.venv/bin/python evals/randomized_tensor_product_parity.py \
  --cases 1000 --seed 20260727 --max-l 6 --max-mul 4 \
  --torch-python evals/.venv-torch/bin/python \
  --mlx-python .venv/bin/python
```

The command exits nonzero if any case fails. Use `--allow-failures` only for
exploratory runs where retaining the complete report is more important than
the exit status.

## Results and replay

Each run creates:

```text
evals/results/tensor-product-randomized/<timestamp>/
├── cases.json
├── torch.json
├── mlx.json
├── report.json
├── report.md
└── failures/
    └── <case-id>.json
```

The report records maximum absolute, maximum elementwise relative, and
norm-scaled errors for every MLX execution variant. It also records whether the
custom path was eligible and the selected kernel kind. A custom-kernel request
that falls back remains visible rather than being mislabeled as a kernel run.

Replay a saved failure with:

```bash
.venv/bin/python evals/randomized_tensor_product_parity.py \
  --replay evals/results/tensor-product-randomized/<run>/failures/<case>.json
```

The random harness is not a replacement for deterministic edge-case tests.
Constructor rejection policies, exact dispatch boundaries, and known
regressions remain explicit unit tests so a generator change cannot
accidentally remove their coverage.
