# Compatibility and numerical conventions

This document defines the numerical and API guarantees of the current
implementation. It is intentionally narrower than full `e3nn` parity.

## Reference implementation

Golden reference data is generated with `e3nn==0.5.8`, `torch==2.7.1`, and
`numpy==2.3.1`.  The generator records the installed versions in every fixture.
Changing a pinned version requires regenerating the fixtures and reviewing all
numerical differences.

## Supported representations

- Integer angular momenta `0 <= l <= 6` are guaranteed by the compatibility
  baseline.
- Pure-Python table and structural tests exercise `0 <= l <= 8` where practical.
- O(3) parity is `p in {-1, 1}`.  The `y` token means `p = (-1)**l`.
- The real basis, component order, phases, Clebsch--Gordan coefficients, and
  Wigner matrices follow upstream `e3nn`.

## Rotation convention

- Euler angles use upstream e3nn's active YXY convention
  `R = Ry(alpha) Rx(beta) Ry(gamma)`.
- Vectors stored in a final dimension transform as `x @ R.T`.
- Wigner matrices act as `D @ x` on column components, equivalently
  `x @ D.T` for arrays with components in the final dimension.
- Proper rotations have determinant `+1`.  For an improper O(3) matrix, the
  discrete inversion factor is applied according to the irrep parity.

## Shapes and broadcasting

- An irreps array has shape `(..., irreps.dim)`.
- Rotation matrices have shape `(..., 3, 3)` and Wigner matrices have shape
  `(..., 2*l+1, 2*l+1)`.
- Leading input and unshared-weight dimensions follow NumPy broadcasting.
- Tensor-product shared weights have shape `(weight_numel,)`; unshared weights
  have shape `(..., weight_numel)`.

## Dtypes

- The compatibility baseline guarantees `float32` numerical execution.
- Static coefficients are generated using Python double precision and converted
  to the input dtype at the MLX boundary.
- Tensor-product inputs must use the same dtype. Mixed dtypes are rejected rather
  than applying implicit promotion.

## Normalization

Spherical harmonics follow upstream `e3nn`:

- `component`: the mean squared value of every component on the sphere is one;
  consequently `sum_m Y_lm(x)^2 = 2*l+1` on the unit sphere.
- `norm`: `sum_m Y_lm(x)^2 = 1` on the unit sphere.
- `integral`: `sum_m Y_lm(x)^2 = (2*l+1)/(4*pi)` on the unit sphere.

Tensor products support `component`, `norm`, and `none` irrep normalization,
and `element`, `path`, and `none` path normalization.  The legacy path token
`component` is accepted as an alias for `element`.

## Zero vectors

With `normalize=True`, spherical harmonics return the normalized scalar for
`l=0` and zeros for `l>0` at the zero vector.  This avoids NaNs while making the
undefined direction explicit.  With `normalize=False`, each degree is a
homogeneous polynomial and therefore naturally vanishes at zero for `l>0`.

## Import policy

`e3nn_core` remains importable without MLX.  `e3nn_mlx` keeps MLX imports lazy
at package import time; invoking a numerical operation requires a working MLX
runtime.

## Release gate

The required Apple-Silicon CI job must run all tests marked `mlx`; a skipped MLX
test is a failure in that job.  Local and non-Apple jobs may skip those tests.
