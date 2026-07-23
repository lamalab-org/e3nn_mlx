# First equivariant operation

This example verifies that a learned linear layer commutes with rotations.

```python
import mlx.core as mx
from e3nn_mlx import o3

mx.random.seed(0)

layer = o3.Linear("4x0e + 3x1o", "8x0e + 2x1o")
x = mx.random.normal((100, layer.irreps_in.dim))
y = layer(x)

R = o3.rand_matrix()
D_in = o3.irreps_wigner_d_from_matrix(layer.irreps_in, R)
D_out = o3.irreps_wigner_d_from_matrix(layer.irreps_out, R)

y_after_input_rotation = layer(x @ D_in.T)
rotated_y = y @ D_out.T
mx.eval(y_after_input_rotation, rotated_y)

error = mx.max(mx.abs(y_after_input_rotation - rotated_y)).item()
print(f"maximum equivariance error: {error:.3e}")
```

The weights may freely mix copies of the same irrep, but cannot mix a scalar
with a vector. Consequently, learning changes the channel basis without
breaking the transformation law.

## Add a nonlinearity

Componentwise nonlinearities preserve equivariance for even scalars, but not
for general vectors. e3nn architectures commonly use a
{class}`~e3nn_mlx.nn.Gate`: scalar channels pass through scalar activations and
then gate non-scalar channels.

```python
from e3nn_mlx import nn

gate = nn.Gate(
    "8x0e", [mx.tanh],
    "2x0e", [mx.sigmoid],
    "2x1o",
)
features = mx.random.normal((100, gate.irreps_in.dim))
activated = gate(features)
```
