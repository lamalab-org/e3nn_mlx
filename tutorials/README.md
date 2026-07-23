# Tutorial compatibility helpers

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

The helpers implement the tutorial's `.Rs`, `to_irrep_transformation`,
`to_irrep_tensor`, `from_irrep_tensor`, `from_geometry`, and `plot` behavior.
They use the current e3nn basis and return MLX arrays. Convert plotting outputs
with `numpy.asarray` instead of calling Torch's `.numpy()` method:

```python
import numpy as np

r, f = SphericalTensor(coefficients).plot(relu=False, res=50)
r, f = np.asarray(r), np.asarray(f)
```

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
