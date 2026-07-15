# e3nn-mlx

Incremental refactor of e3nn into a backend-neutral core plus an MLX-native runtime.

The P0 implementation currently provides:

- backend-agnostic O(3) metadata in `e3nn_core`
- e3nn-compatible real Wigner-3j and Clebsch--Gordan coefficients
- batched real Wigner matrices and O(3) parity transforms through arbitrary `l`
- spherical harmonics generated recursively from the canonical Wigner basis
- weighted and unweighted MLX tensor products and standard wrappers
- lightweight MLX Linear, Gate, Norm, and reduction operations
- pinned upstream e3nn numerical fixtures and Apple-Silicon CI coverage

P0 guarantees numerical compatibility through `l=6`; reference and structural
tests exercise selected operations through `l=8`.  See
[`docs/P0_COMPATIBILITY.md`](docs/P0_COMPATIBILITY.md) for conventions, shapes,
normalization, and the release gate.

## Development

Install the MLX and test extras and run the suite on Apple Silicon:

```bash
python -m pip install -e '.[mlx,test]'
python -m pytest
```

MLX tests may skip when Metal is unavailable locally.  The required macOS CI job
sets `E3NN_MLX_REQUIRE_RUNTIME=1`, which turns such skips into failures.
