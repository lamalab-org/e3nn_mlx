# Reference fixture generation

The Torch parity suite deliberately does not import Torch in the normal MLX
test environment. Instead, deterministic reference arrays are generated with
the exact versions pinned in `requirements.txt` and committed to
`tests/reference_data/`.

Create an isolated environment, install `requirements.txt`, and generate the
general e3nn fixtures with:

```bash
python tests/reference_generation/generate_e3nn_reference.py
```

Generate the comprehensive Torch numerical-parity fixture with:

```bash
python tests/reference_generation/generate_torch_parity.py
```

`torch_parity_reference.npz` contains deterministic inputs, identical
parameters, outputs, gradients, complete Hessians, rotated outputs, and scalar
equivariance errors. `torch_parity_manifest.json` records the versions,
representations, instruction definitions, tensor shapes, and parameter names.
The coverage includes:

- all TensorProduct connection modes, shared and per-sample weights, plus the
  FullTensorProduct, FullyConnectedTensorProduct, ElementwiseTensorProduct,
  and TensorSquare wrappers;
- Linear, Norm, Activation, Gate, NormActivation, Identity, Extract,
  ExtractIr, Dropout, BatchNorm, FullyConnectedNet, S2Activation, and
  SO3Activation;
- input and parameter gradients, full Hessians, and the mixed
  input/parameter Hessian blocks;
- gate-points and v2106 network outputs under identical named parameters;
- Torch and MLX equivariance residuals evaluated with the same fixed rotation.

Run the consumer suite with:

```bash
E3NN_MLX_REQUIRE_RUNTIME=1 python -m pytest -q \
  tests/test_torch_numerical_parity.py
```

Fixture updates must be reviewed; never regenerate them implicitly as part of
the test suite.
