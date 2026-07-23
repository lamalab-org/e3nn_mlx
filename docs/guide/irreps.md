# Irreducible representations

An equivariant feature is not described only by its array shape. It also needs
a rule for how its components transform under rotations and inversion.
`Irrep` describes one irreducible representation of $O(3)$ and `Irreps`
describes a direct sum of them.

```python
from e3nn_mlx import o3

scalar = o3.Irrep("0e")
vector = o3.Irrep("1o")
features = o3.Irreps("16x0e + 8x1o")

assert scalar.dim == 1
assert vector.dim == 3
assert features.dim == 40
```

The integer $l$ determines the component count $2l+1$. The parity is `e` for
even and `o` for odd under inversion. Ordinary polar vectors are `1o`;
pseudovectors are `1e`.

## Typed and raw arrays

The high-level `o3` and `nn` namespaces accept raw MLX arrays, matching the
style of upstream e3nn:

```python
import mlx.core as mx
from e3nn_mlx import o3

linear = o3.Linear("2x0e + 1x1o", "3x0e + 2x1o")
x = mx.zeros((32, linear.irreps_in.dim))
y = linear(x)
```

Use `IrrepsArray` to carry the representation together with its data and catch
mismatches at operation boundaries:

```python
from e3nn_mlx import IrrepsArray, Linear

x_typed = IrrepsArray("2x0e + 1x1o", x)
y_typed = Linear(x_typed.irreps, "3x0e + 2x1o")(x_typed)
assert y_typed.irreps == o3.Irreps("3x0e + 2x1o")
```

## Rotating features

For an irrep specification $\rho$, `irreps_wigner_d` constructs the block
matrix $D_\rho(R)$. A feature transforms as $x' = xD_\rho(R)^T$ when features
occupy the last array axis.

```python
R = o3.rand_matrix()
D = o3.irreps_wigner_d_from_matrix(features, R)
feature_values = mx.zeros((32, features.dim))
x_rotated = feature_values @ D.T
```

This transformation is the basis of the equivariance checks described in the
[equivariance guide](equivariance.md).
