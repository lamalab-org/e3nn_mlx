# Equivariant graph convolution

For nodes $i$ and neighbors $j$, a common convolution is

$$
f'_i = \frac{1}{\sqrt{z}}
\sum_{j \in \mathcal{N}(i)}
f_j \otimes_{w(\lVert r_{ij}\rVert)} Y(r_{ij}),
$$

where spherical harmonics encode edge directions, a radial network produces
tensor-product weights, and scatter-sum aggregates messages.

The following sketch exposes all important array shapes:

```python
import mlx.core as mx
from e3nn_mlx import math, o3, scatter_sum

irreps_node = o3.Irreps("8x0e + 4x1o")
irreps_sh = o3.Irreps.spherical_harmonics(2)

tp = o3.FullyConnectedTensorProduct(
    irreps_node,
    irreps_sh,
    irreps_node,
    internal_weights=False,
    shared_weights=False,
)

# positions: (nodes, 3); edge_src/edge_dst: (edges,)
edge_vec = positions[edge_dst] - positions[edge_src]
edge_length = mx.linalg.norm(edge_vec, axis=-1)
edge_attr = o3.spherical_harmonics(
    irreps_sh, edge_vec, normalize=True, normalization="component"
)

radial = math.soft_one_hot_linspace(
    edge_length,
    start=0.0,
    end=max_radius,
    number=num_basis,
    basis="smooth_finite",
    cutoff=True,
)
weights = radial_network(radial)  # (edges, tp.weight_numel)
messages = tp(node_features[edge_src], edge_attr, weights)
output = scatter_sum(messages, edge_dst, positions.shape[0])
```

Distance-derived weights are rotation invariant; spherical harmonics and the
tensor product carry the directional transformation law. Summation is
equivariant because it combines only features with the same representation.

For a complete trainable implementation, use
{class}`~e3nn_mlx.models.v2106.Convolution` or inspect its source.
