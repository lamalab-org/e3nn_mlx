# Changelog

All notable changes to `e3nn-mlx` will be documented in this file. Versions
follow [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- Expose the array-returning methods upstream e3nn defines on `Irrep` and
  `Irreps`: `D_from_angles`, `D_from_matrix`, `D_from_quaternion`,
  `D_from_axis_angle`, plus `Irreps.randn` and `Irreps.index`. They dispatch
  through a new `e3nn_core.runtime` hook registry that `e3nn_mlx` populates on
  import, so `e3nn_core` still imports and computes metadata with no array
  library installed.
- Support half-integer `j` in `su2_clebsch_gordan`, which `su2_generators`
  already accepted.

### Changed

- `Irreps.__str__` now always prints the multiplicity, so `Irreps("1o+2x0e")`
  renders as `1x1o+2x0e` exactly like upstream. Parsing is unchanged and still
  accepts either spelling.
- `Irrep.iterator` yields the natural parity of each degree first
  (`0e, 0o, 1o, 1e, ...`), matching upstream's documented order.
- `Irreps.lmax` raises `ValueError` when no part has a nonzero multiplicity
  instead of returning `-1`, matching upstream.
- `clebsch_gordan` raises on O(3)-forbidden couplings such as `(1o, 1o, 1o)`
  rather than silently returning the SO(3) coefficients, since unlike
  `wigner_3j` its arguments carry a parity.
- `so3_generators` evaluates its basis change as two matrix products instead of
  a quadruple loop. Output is bitwise identical; cold-start cost for `l=0..11`
  drops from roughly 154 ms to 13 ms.

### Removed

- **Breaking:** `RotationAngles`, `WignerDKey`, and `Wigner3jKey`. They were
  exported but never constructed anywhere, had no upstream e3nn analogue, and
  `WignerDKey.irrep` hardcoded even parity so it could not address odd irreps
  at all.

### Fixed

- `MulIrrep` now unpacks as `mul, ir`, so upstream's pervasive
  `for mul, ir in irreps` idiom works instead of raising `TypeError`.
- `WignerDKey`/`RotationAngles` labelled e3nn's Euler convention `"zyz"`; it is
  the active intrinsic YXY convention, as the rest of the tree already stated.
- Leave the normalization coefficient undivided when the path-normalization
  divisor is zero, as upstream does, instead of collapsing the path weight to
  zero. Reachable with mode `u<vw` at multiplicity 1.

- Match upstream e3nn's input-major default `Linear` instruction and flattened
  weight order. This preserves the meaning of external weights when input and
  output irreps contain repeated compatible blocks.
- Preserve external-weight gradients in the grouped `Linear` execution path.
- Accumulate repeated unweighted tensor-product instructions into their
  declared output blocks and broadcast unshared weights over all compatible
  leading dimensions.
- Validate tensor-product weight dtypes, unshared-weight batch dimensions,
  invalid unweighted `uvw` instructions, and instructions that reference
  zero-multiplicity irreps consistently across execution paths.
- Keep shared TensorProduct weights compact during batched contractions,
  avoiding per-item parameter-gradient materialization, and route dense scalar
  Metal plans through the faster general MLX path at their measured crossover.

### Testing

- Add seeded, replayable Torch/e3nn-versus-MLX randomized parity harnesses for
  tensor products and the major numerical operation families. The harnesses
  compare identical NumPy-generated inputs, forward values, and VJPs.
- Consolidate performance evaluation into one three-way Torch-CPU,
  general-MLX, and kernel-MLX suite with smoke and full presets.
- Add matched end-to-end benchmarks for gate-points 2102, v2106 simple, and
  v2106 attributed networks on deterministic fixed graph topology.
- Record the actually selected MLX execution path in performance artifacts so
  automatic kernel fallbacks remain distinguishable from kernel dispatches.

### Documentation

- Document generated-kernel shape, dtype, dispatch, differentiation, model,
  fusion, and startup-cost limitations.

## 0.1.0

Initial alpha release with:

- backend-neutral irreps, rotations, Wigner coefficients, and instructions;
- MLX spherical harmonics, tensor products, linear and gated operations;
- generated Metal kernels with differentiable general-MLX fallbacks;
- e3nn-style `o3`, `nn`, `math`, and point-model namespaces;
- gate-points 2102 and modular v2106 model implementations;
- Apple Silicon and Linux MLX validation.
