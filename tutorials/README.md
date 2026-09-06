# Tutorial compatibility helpers


## Requirements

The notebooks need extras that the base package does not install:

```bash
python -m pip install -e '.[tutorials]'
```

## Where these tutorials come from

- `data_types.ipynb` and `operations_on_spherical_tensors.ipynb` adapt
  [Tess E. Smidt's early e3nn tutorials](https://blondegeek.github.io/e3nn_tutorial/).
  They introduce irreducible representations and real spherical harmonics,
  show how Cartesian tensors decompose into irreps, and build intuition for
  rotations, addition, inner products, and Clebsch--Gordan tensor products.
- `e3nn_MRS_invariants_tutorial_lecture_pt1.ipynb` adapts Martin Uhrin and
  Thomas Hardin's 2021 tutorial,
  [Analysing Geometry and Structure of Atomic Configurations with Equivariant and Invariant Functions](https://colab.research.google.com/drive/1duR1Y-roE_CSL3hrGxINMZ4XIepHvoHt)
  ([lecture video](https://www.youtube.com/watch?v=6W3syIWfOA4)). It develops
  permutation-, translation-, and rotation-aware descriptions of atomic
  structures, then constructs invariant power-spectrum and bispectrum
  features from spherical expansions and tensor products.

These notebooks were written against older Torch/e3nn interfaces. The copies
in this directory preserve their teaching flow while replacing the numerical
operations with MLX and e3nn-mlx.

The `data_types.ipynb` notebook was written for an early e3nn API that exposed
`e3nn.rs` and `e3nn.tensor`. Use the local MLX-native helpers instead:

```python
import mlx.core as mx
from e3nn_mlx import o3
from tutorials import CartesianTensor, IrrepTensor, SphericalTensor
```

When the notebook server starts inside this directory, the equivalent import
is:

```python
from tensor_helpers import CartesianTensor, IrrepTensor, SphericalTensor
```

These are compatibility helpers, not an independent tensor engine.
`IrrepTensor` and `SphericalTensor` inherit from
`e3nn_mlx.IrrepsArray`, so they can be passed directly to the library's
representation-aware operations. `CartesianTensor` keeps a Cartesian value
until conversion and exposes its `e3nn_mlx.o3.ReducedTensorProducts` operator
as `decomposition`; `to_irrep_tensor()` returns an `IrrepsArray` subclass.

The helpers implement the tutorial's `.Rs`, `to_irrep_transformation`,
`to_irrep_tensor`, `from_irrep_tensor`, `from_geometry`, and `plot` behavior.
They use the current e3nn basis and return MLX arrays. Convert plotting outputs
with `numpy.asarray` instead of calling Torch's `.numpy()` method:

```python
import numpy as np

r, f = SphericalTensor(coefficients).plot(relu=False, res=50)
r, f = np.asarray(r), np.asarray(f)
```

The historical explicit-bandwidth constructor remains supported:

```python
spherical = SphericalTensor(signal, L_max)
```

The second argument is optional because the helper can infer `L_max` from the
coefficient count.

The later invariants tutorial's representation-descriptor form and weighted
Dirac projection are supported as well:

```python
descriptor = SphericalTensor(lmax=4, p_val=1, p_arg=-1)
coefficients = descriptor.sum_of_diracs(positions, values)
peaks = descriptor.with_peaks_at(positions, values)
rotation = descriptor.D_from_angles(alpha, beta, gamma)
```

These methods return MLX arrays. Plotly widget properties do not accept MLX
arrays directly; convert both initial data and interactive updates with
`numpy.asarray`:

```python
widget.data[0].z = np.asarray(calc(positions)).T
```

The core `o3.ReducedTensorProducts` API also accepts the historical
`filter_ir_mid` and `filter_ir_out` keyword arguments used to build filtered
bispectrum contractions.

Spherical tensors can be added even when their maximum degrees differ; the
lower-bandwidth operand is zero-padded first. The historical `.signal` name is
also available as an alias for the MLX coefficient array, and `.dot()` computes
the coefficient-space inner product used by the tutorial. The `@` operator
performs a full tensor product and regroups polar and axial copies by degree,
matching the tutorial's SO(3)-only output and slice ordering.

As in the original tutorial, `plot()` defaults to unscaled,
integral-normalized spherical harmonics. The optional `"component"` and
`"norm"` normalization modes use the modern e3nn scaling conventions.
Likewise, `from_geometry()` defaults to the tutorial's adjusted least-squares
projection instead of a raw sum of spherical-harmonic coefficients.

`CartesianTensor` assumes ordinary `(x, y, z)` Cartesian axes. The historical
notebook's manual `(y, z, x)` permutation belongs to its older e3nn basis and
must not be repeated with these helpers. To reproduce the historical
orientation only for visualization, request it explicitly:

```python
r, f = SphericalTensor(coefficients).plot(
    relu=False,
    radius=True,
    res=50,
    legacy_axes=True,
)
```

The old rank-two Cartesian table additionally used historical Clebsch--Gordan
phases. Reproduce its transformation matrix with:

```python
cartesian = CartesianTensor(matrix, legacy_basis=True)
Rs, Q = cartesian.to_irrep_transformation()
```

This mode is intentionally separate from `SphericalTensor.plot(
legacy_axes=True)`: `legacy_basis` changes coefficients and the
change-of-basis matrix, while `legacy_axes` changes only displayed coordinates.
