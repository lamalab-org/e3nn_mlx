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

## Intentional MVP Deviations

- No attempt at full PyTorch parity in the first slice.
- Numerical Wigner/CG tables are deferred; only symbolic metadata lands now.
- `e3nn_mlx` imports MLX lazily so the package can be imported in environments where MLX
  is not installed.
- Tests use `unittest` for now because the workspace does not yet have `pytest`.

## Immediate Next Steps

1. Finish static metadata and protocol seams.
2. Add `IrrepsArray` with deterministic chunk metadata.
3. Implement rotations/Wigner helpers on top of MLX.
4. Add spherical harmonics and tensor product, symbolic first and numeric second.
