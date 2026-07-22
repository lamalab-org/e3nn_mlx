# Testing equivariance

An operation $f : \rho_{in} \rightarrow \rho_{out}$ is equivariant when

$$
f(D_{in}(R)x) = D_{out}(R)f(x)
$$

for every $R \in O(3)$. Numerical tests sample transformations and compare the
two sides. For a linear map:

```python
import mlx.core as mx
from e3nn_mlx import o3

layer = o3.Linear("3x0e + 2x1o", "4x0e + 3x1o")
x = mx.random.normal((16, layer.irreps_in.dim))
R = o3.rand_matrix()
D_in = o3.irreps_wigner_d_from_matrix(layer.irreps_in, R)
D_out = o3.irreps_wigner_d_from_matrix(layer.irreps_out, R)

actual = layer(x @ D_in.T)
expected = layer(x) @ D_out.T
mx.eval(actual, expected)

assert mx.allclose(actual, expected, rtol=2e-4, atol=2e-5).item()
```

## Coordinate-dependent operations

For spherical harmonics or point models, rotate coordinates and every feature
according to its own irreps. Graph connectivity must remain the same, or be
recomputed from rotation-invariant distances. Scalar energies should remain
unchanged; vector forces should rotate with the coordinates.

## What to test

An effective model test covers:

1. several random proper rotations;
2. inversion when the model claims $O(3)$ rather than only $SO(3)$ symmetry;
3. batches and nontrivial multiplicities;
4. gradients, when forces or response properties are model outputs;
5. both generated-kernel and general-MLX execution paths.

Passing one rotation is not a proof, but randomized property tests catch
incorrect parity, layout, normalization, and aggregation behavior efficiently.
