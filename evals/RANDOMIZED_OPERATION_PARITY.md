# Randomized parity for major operations

`randomized_operation_parity.py` complements the deeper randomized
TensorProduct qualification with seeded comparisons across the rest of the
important numerical API.

For every case, isolated Torch/e3nn and MLX workers receive identical
NumPy-generated float32 values. The harness compares both the forward output
and a VJP formed with the same deterministic, nonuniform cotangent.

## Covered operations

- Euler rotation matrices and Wigner-D matrices
- spherical harmonics, including general and custom-kernel requests
- Linear, including eager/compiled MLX execution, repeated irrep blocks,
  external weights, normalization, and biases
- Norm, Activation, Gate, and NormActivation
- BatchNorm in evaluation mode with identical affine parameters and running
  statistics
- S2Activation and SO3Activation
- radial bases and the smooth unit step
- ReducedTensorProducts through its gauge-invariant projected tensor
- scatter aggregation through general and custom-kernel requests
- FullTensorProduct, FullyConnectedTensorProduct,
  ElementwiseTensorProduct, and TensorSquare

Reduced tensor-product coefficient channels cannot generally be compared
element by element because an orthonormal reduced basis is defined only up to
sign, permutation, and orthogonal mixing inside repeated irreps. The harness
therefore projects the coefficients back into the original tensor space before
comparing outputs and VJPs.

Whole graph networks and MACE models retain deterministic fixed-topology
parity tests with identical named parameters. Randomizing entire architectures
would combine operator correctness with parameter-conversion semantics and is
kept separate from this operator harness.

## Running it

Run a quick check:

```bash
.venv/bin/python evals/randomized_operation_parity.py \
  --cases-per-kind 5 --seed 20260727 --max-l 4
```

Run a larger qualification:

```bash
.venv/bin/python evals/randomized_operation_parity.py \
  --cases-per-kind 100 --seed 20260727 --max-l 6 \
  --torch-python evals/.venv-torch/bin/python \
  --mlx-python .venv/bin/python
```

Select individual operation families by repeating `--kind`:

```bash
.venv/bin/python evals/randomized_operation_parity.py \
  --cases-per-kind 250 \
  --kind spherical_harmonics \
  --kind linear \
  --kind gate
```

Results are written to:

```text
evals/results/operation-randomized/<timestamp>/
├── cases.json
├── torch.json
├── mlx.json
├── report.json
├── report.md
└── failures/
```

Each failure JSON includes the complete case and both worker results. Replay it
with:

```bash
.venv/bin/python evals/randomized_operation_parity.py \
  --replay evals/results/operation-randomized/<run>/failures/<case>.json
```

The command exits nonzero on a discrepancy. `--allow-failures` keeps a
discovery run successful while retaining all failure artifacts.
