# Upstream `tests/nn` compatibility matrix

## Reference snapshot and scope

This port is audited against `e3nn/e3nn` commit
[`2aa7f58440a06b15352a2cbce01fa4c26f824969`](https://github.com/e3nn/e3nn/tree/2aa7f58440a06b15352a2cbce01fa4c26f824969/tests/nn),
dated 2026-02-13.

The requested scope is every Python test directly under `tests/nn`. The following model tests are intentionally excluded:

- `tests/nn/models/gate_points_2101_test.py`
- `tests/nn/models/gate_points_2102_test.py`
- `tests/nn/models/v2203/sparse_voxel_convolution_test.py`

The tests are semantic MLX ports rather than textual Torch translations:

- representation-carrying values use `IrrepsArray` where the MLX API requires representation metadata;
- TorchScript and `torch.compile` checks use `mlx.core.compile`;
- Torch autograd checks use MLX `grad` or `mlx.nn.value_and_grad`;
- parameter checks use MLX module parameter trees;
- `copy.deepcopy` checks are retained;
- upstream's session-wide float64 repetitions are not applicable because the
  current MLX runtime contract is float32.

## Requirement mapping

| Upstream file | Upstream cases | MLX evidence |
|---|---|---|
| `activation_test.py` | All 3 activation/irrep combinations; normalized second moment; constant activation scale; proper and improper equivariance; compilation | `tests/test_upstream_nn_basic.py` |
| `batchnorm_test.py` | Train/eval equivariance; all 16 affine/reduce/normalization/instance modes; both instance modes with norm and component output statistics | `tests/test_upstream_nn_norm_fc.py` |
| `dropout_test.py` | Evaluation identity; training mask values; irrep-component-shared masks; proper and improper equivariance under a fixed random seed; compilation; deepcopy | `tests/test_upstream_nn_basic.py` |
| `extract_test.py` | Multiple outputs; both `squeeze_out` modes; `ExtractIr`; proper and improper equivariance; compilation of multiple and single outputs; deepcopy | `tests/test_upstream_nn_basic.py` |
| `fc_test.py` | All 8 activation/variance/output-activation combinations with the exact `(1000, 500, 1500, 4)` widths and 2,000 inputs; upstream variance interval; compilation; MLX gradients and parameter tree | `tests/test_upstream_nn_norm_fc.py` |
| `gate_test.py` | Exact standalone two-output `_Sortcut`; five-argument Gate; normalized output; proper and improper equivariance; compilation; parameterlessness | `tests/test_upstream_nn_gate.py` |
| `normact_test.py` | Both bias modes and both nonlinearities; scalar values; vector norms and directions; zero preservation; finite zero gradients; broad-irrep proper and improper equivariance; compilation; parameter tree | `tests/test_upstream_nn_norm_fc.py` |
| `s2act_test.py` | All 16 activation/normalization/value-parity/argument-parity combinations; 10 trials each for proper rotations and inversion; random grid rotations; `lmax_in=3`, `lmax_out=6`, resolution 120 | `tests/test_upstream_nn_s2act.py` |
| `so3act_test.py` | All 8 activation/`lmax` equivariance combinations with 10 proper/improper trials; all 4 compiled l=5 identity cases and aspect ratios | `tests/test_upstream_nn_so3act.py` |

`fc_test.py::test_data_parallel` is guarded upstream by `torch.cuda.is_available()` and is skipped on non-CUDA systems.
MLX does not expose the Torch CUDA `DataParallel` API, so that wrapper-specific assertion is backend-inapplicable. The
underlying requirements that the network exposes trainable parameters and supports backward differentiation are covered by
the MLX parameter-tree and `value_and_grad` assertions in the eight FC cases.

## Implementation supplied by this port

- parity-aware, second-moment-normalized `Activation`;
- irrep-copy-shared equivariant `Dropout`;
- representation-aware `Extract` and `ExtractIr`;
- upstream-compatible five-argument `Gate` and `_Sortcut`;
- equivariant `BatchNorm` and `NormActivation`;
- variance-normalized `FullyConnectedNet`;
- parity-aware `S2Activation`, including input-bandwidth-aware S2 projection normalization;
- dense Wigner-basis `SO3Grid` and `SO3Activation` with normalized Haar quadrature.

## Resolved compatibility defects

Every item below is fixed and protected by the mapped regression suites:

1. S2 output projection rejected the valid upstream case where `lmax_out > lmax_in` and used the wrong bandwidth normalization.
2. Gate lacked the upstream five-argument API and sorted extraction path.
3. FC layer construction incorrectly used strict adjacent-width zipping.
4. Dropout needed one random mask per irrep copy, shared over all components, to preserve equivariance.
5. Odd scalar activations and spherical signals required activation-parity propagation to preserve inversion equivariance.
6. Norm-based activation needed epsilon clamping before the square root to keep zero-input gradients finite.

## Release gate

Completion requires all of the following from the repository root on Apple Silicon:

```bash
E3NN_MLX_REQUIRE_RUNTIME=1 .venv/bin/python -m pytest -q tests/test_upstream_nn_basic.py tests/test_upstream_nn_gate.py tests/test_upstream_nn_norm_fc.py tests/test_upstream_nn_s2act.py tests/test_upstream_nn_so3act.py
E3NN_MLX_REQUIRE_RUNTIME=1 .venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q -m 'not mlx'
```

The runtime-required command turns any skipped MLX test into a failure.
