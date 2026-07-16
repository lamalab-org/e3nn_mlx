# Porting Map

This document anchors the first migration slice and records the intended ownership split.

## Upstream Sources

- `e3nn` contributes the O(3) symbolic metadata layer: irreps, parity, selection rules,
  tensor-product path metadata, normalization rules, and Wigner/Clebsch-Gordan bookkeeping.
- `e3nn-jax` contributes the array-front-end shape: `IrrepsArray`, symbolic/numeric split
  for higher-level operators, and a backend-oriented design that does not assume PyTorch.

## New Package Split

- `e3nn_core`
  - Pure Python immutable metadata only.
  - No tensor library imports.
  - Owns irreps parsing, regrouping, selection rules, tensor-product instruction generation,
    normalization metadata, and rotation/Wigner bookkeeping keys.
- `e3nn_backend`
  - Thin protocol for array backends.
  - Runtime registry for selecting a backend implementation.
- `e3nn_mlx`
  - MLX-specific runtime.
  - Owns arrays, transforms, compilation/fusion hooks, and future extension seams.

## Planned API Ownership

1. `e3nn_core.irreps`
   - `Irrep`
   - `MulIrrep`
   - `Irreps`
   - parsing, formatting, regrouping, sorting, slicing metadata
2. `e3nn_core.instructions`
   - `TensorProductInstruction`
   - symbolic tensor-product path generation
3. `e3nn_core.normalization`
   - `NormalizationMetadata`
   - path and irrep normalization metadata
4. `e3nn_core.cg` / `e3nn_core.wigner`
   - cache keys and book-keeping dataclasses first
   - numerical table generation later if needed
5. `e3nn_mlx.irreps_array`
   - first-class MLX object with deterministic chunking by irreps
6. `e3nn_mlx.ops_*`
   - numerical operators grouped by concern
7. `e3nn_mlx.nn_*`
   - lightweight module wrappers over coarse MLX kernels
8. `e3nn_mlx.o3`, `e3nn_mlx.nn`, and `e3nn_mlx.math`
   - e3nn-shaped high-level namespaces
   - raw MLX array input/output for upstream-style code
   - type-preserving `IrrepsArray` dispatch for representation-aware code
9. `e3nn_mlx.nn.models`
   - upstream-shaped model import paths and raw-array boundary adapters
   - delegates to the canonical implementations in `e3nn_mlx.models`

## Intentional deviations

- `e3nn_mlx` imports MLX lazily so the package can be imported in environments where MLX
  is not installed.
- The high-level namespaces cover implemented functionality rather than claiming full
  upstream PyTorch/e3nn parity.
- `Irreps` remains backend-neutral and therefore does not allocate random MLX arrays.
- The canonical numerical implementations remain in `ops_*`, `nn_*`, and `models`; the
  high-level packages are zero-copy compatibility adapters rather than duplicate kernels.
