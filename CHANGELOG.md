# Changelog

All notable changes to `e3nn-mlx` will be documented in this file. Versions
follow [Semantic Versioning](https://semver.org/).

## Unreleased

### Fixed

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

### Testing

- Add seeded, replayable Torch/e3nn-versus-MLX randomized parity harnesses for
  tensor products and the major numerical operation families. The harnesses
  compare identical NumPy-generated inputs, forward values, and VJPs.
- Consolidate performance evaluation into one three-way Torch-CPU,
  general-MLX, and kernel-MLX suite with smoke and full presets.

## 0.1.0

Initial alpha release with:

- backend-neutral irreps, rotations, Wigner coefficients, and instructions;
- MLX spherical harmonics, tensor products, linear and gated operations;
- generated Metal kernels with differentiable general-MLX fallbacks;
- e3nn-style `o3`, `nn`, `math`, and point-model namespaces;
- gate-points 2102 and modular v2106 model implementations;
- Apple Silicon and Linux MLX validation.
