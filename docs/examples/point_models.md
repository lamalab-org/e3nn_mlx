# Point-cloud models

The library contains the gated `gate_points_2102` network and the modular
`v2106` family. They accept positions, node features, atomic attributes, and
batch indices in the layouts documented by their constructors.

```python
import mlx.core as mx
from e3nn_mlx.models.v2106 import SimpleNetwork

model = SimpleNetwork(
    irreps_in="1x0e",
    irreps_out="1x0e",
    max_radius=3.0,
    num_neighbors=8.0,
    num_nodes=32.0,
    mul=16,
    layers=2,
    lmax=2,
    pool_nodes=True,
)

data = {
    "pos": positions,
    "x": mx.ones((positions.shape[0], 1)),
    "batch": batch,
}
prediction = model(data)
```

## Compiled fixed-edge execution

Radius-graph construction has a dynamic output size and therefore remains an
eager boundary. In training loops, build `edge_src` and `edge_dst` outside the
compiled function and call `forward_with_edges`. The radial embedding,
spherical harmonics, message passing, gates, and reduction can then remain in
the compiled MLX graph.

```python
def forward(pos, features, graph_batch, src, dst):
    return model.forward_with_edges(
        pos,
        features,
        graph_batch,
        src,
        dst,
        num_graphs=1,
    ).array

compiled_forward = mx.compile(forward)
prediction = compiled_forward(
    data["pos"], data["x"], data["batch"], edge_src, edge_dst
)
mx.eval(prediction)
```

Periodic systems require a neighbor list that also supplies cell-shifted edge
vectors. The generic `radius_graph` helper computes Euclidean neighbors for the
given coordinates and does not infer periodic images.
