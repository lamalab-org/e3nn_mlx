# Upstream `tests/o3` compatibility matrix

## Reference snapshot

This port is audited against `e3nn/e3nn` commit
[`2aa7f58440a06b15352a2cbce01fa4c26f824969`](https://github.com/e3nn/e3nn/tree/2aa7f58440a06b15352a2cbce01fa4c26f824969/tests/o3),
dated 2026-02-13. That snapshot contains 84 `test_*` function definitions in 13 test files, plus the
non-test benchmark `experimental/benchmark_pt2.py`.

The tests are semantic MLX ports, not textual Torch translations:

- numerical inputs carrying representations use `IrrepsArray`;
- TorchScript and `torch.compile` assertions use `mlx.core.compile` and the modules' compiled paths;
- Torch autograd assertions use MLX `grad`;
- `.state_dict()` and Torch pickle assertions use MLX parameter trees, `.npz` weight round trips, and deepcopy;
- Torch dtype/device migration has no direct MLX equivalent; dtype preservation is tested at operator boundaries;
- experimental Torch v2 kernels are covered by production-wrapper comparisons against explicit tensor products and
  independent numerical references.

## Requirement mapping

| Upstream file | Upstream tests covered | MLX evidence |
|---|---|---|
| `angular_spherical_harmonics_test.py` | `test_jit`, both angular equivariance/irrep identities, Cartesian equality | `tests/test_upstream_o3_rotations_sh.py` |
| `cartesian_spherical_harmonics_test.py` | unusual degree/irrep calls, invalid parity, zeros, equivariance, autodiff, normalization through `l=10`, closure, parity through `l=11`, recurrence/Jacobian, module compile/state | `tests/test_upstream_o3_rotations_sh.py`, `tests/test_spherical_harmonics.py` |
| `irreps_test.py` | construction, properties, arithmetic, empties, indexing, concatenation, membership, invalid inputs, expected-failure case, multiplicity slicing | `tests/test_upstream_o3_irreps_wigner.py`, `tests/test_irreps.py` |
| `linear_test.py` | equivariance/compile/normalization, per-output biases, disconnected outputs, exact TensorProduct equivalence, masks, explicit/default/empty instructions, shared and unshared views, feature channels | `tests/test_upstream_o3_linear_norm.py`, `tests/test_mlx_module_state.py`, `tests/test_mlx_module_integration.py` |
| `norm_test.py` | empty/random irreps, squared/unsquared equivariance and compile, finite zero gradient, vector values | `tests/test_upstream_o3_linear_norm.py`, `tests/test_modules.py` |
| `reduce_tensor_test.py` | reconstruction/state, rank-2 antisymmetry, Levi-Civita, rank-3 `l=2` antisymmetry, elasticity symmetry and parity | `tests/test_upstream_o3_reduce_tensor.py` |
| `rotation_test.py` | coordinate conversion, all representation conversion cycles, composition, inverse, Haar axis-angle isotropy, fixed-axis matrices | `tests/test_upstream_o3_rotations_sh.py`, `tests/test_rotations.py` |
| `s2_test.py` | all 111 valid grid-projection combinations, all 111 coefficient round trips, eight nonlinear equivariance configurations | `tests/test_upstream_o3_s2.py` |
| `tensor_product_sub_test.py` | fully connected and split normalization, Identity, full product, Norm, TensorSquare component/norm/none behavior and elasticity irreps | `tests/test_upstream_o3_tensor_product.py`, `tests/test_tensor_product_wrappers.py` |
| `tensor_product_test.py` | all connection modes, bilinearity, `right`, variance, equivariance, weight linearity, element/path normalization, empties, eager/compiled equality, shared/unshared flat and list weights, broadcasting, validation, views, deepcopy/save/load, triangular mode | `tests/test_upstream_o3_tensor_product.py`, `tests/test_tensor_product.py`, `tests/test_tensor_product_correctness.py`, `tests/test_p0_equivariance.py` |
| `wigner_test.py` | all permutation symmetries, rotation invariance, Cartesian `l=1`, integer and half-integer SU(2) algebra | `tests/test_upstream_o3_irreps_wigner.py`, `tests/test_cg_and_wigner_core.py` |
| `experimental/test_elementwise_tp.py` | production elementwise wrapper agrees with explicit/independent paths for both split patterns and compiles | `tests/test_upstream_o3_tensor_product.py`, `tests/test_tensor_product_correctness.py` |
| `experimental/test_fulltp.py` | production full wrapper agrees with explicit/independent paths for all four input combinations and compiles | `tests/test_upstream_o3_tensor_product.py`, `tests/test_tensor_product_correctness.py` |

## Release gate

Completion requires both of the following from the repository root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q -m 'not mlx'
```

On Apple Silicon, the first command must execute every `mlx` test without skips. The second command verifies that the
backend-neutral core and lazy-import policy remain valid without a working Metal runtime.
