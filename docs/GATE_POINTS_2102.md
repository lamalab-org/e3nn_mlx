# `gate_points_2102` MLX model

## Scope

This is a semantic MLX port of upstream
[`e3nn.nn.models.gate_points_2102`](https://github.com/e3nn/e3nn/blob/2aa7f58440a06b15352a2cbce01fa4c26f824969/e3nn/nn/models/gate_points_2102.py).
It includes:

- normalized radial bases (`gaussian`, `cosine`, `smooth_finite`, `fourier`, and `bessel`);
- smooth unit-step and cosine cutoff functions;
- indexed scatter-sum with compiled execution and gradients;
- eager, directed, loop-free, batched radius graphs;
- the per-edge radial `Convolution`;
- the complete gated `Network`, including optional node inputs and attributes and optional graph reduction.

## Public API

```python
import e3nn_mlx as e3nn
from e3nn_mlx.models.gate_points_2102 import Network

network = Network(
    irreps_in="3x0e + 2x1o",
    irreps_hidden="5x0e + 5x0o + 5x1e + 5x1o",
    irreps_out="2x0o + 2x1o + 2x2e",
    irreps_node_attr="10x0e",
    irreps_edge_attr=e3nn.Irreps.spherical_harmonics(3),
    layers=3,
    max_radius=2.0,
    number_of_basis=5,
    radial_layers=2,
    radial_neurons=100,
    num_neighbors=4.0,
    num_nodes=5.0,
)

output = network({"pos": positions, "x": features, "z": attributes, "batch": batch})
```

`output` is an `IrrepsArray` carrying `network.irreps_out`.

## Compilation boundary

MLX 0.31 provides differentiable indexed addition but no device-side dynamic `nonzero`. Consequently,
`radius_graph` synchronizes once to construct its variable-length integer edge list. This discrete topology operation is not
differentiable in any graph library and is intentionally outside compiled model execution.

For a fixed topology, compile `forward_with_edges`:

```python
edges = e3nn.radius_graph(positions, network.max_radius, batch)

def forward(pos, x, z, batch, src, dst):
    return network.forward_with_edges(
        pos, x, z, batch, src, dst, num_graphs=number_of_graphs
    ).array

compiled_forward = mx.compile(forward)
```

Position gradients still flow through edge vectors, lengths, radial bases, cutoffs, spherical harmonics, and every
convolution. Only the discrete choice of which edges exist is held fixed.

## Verification evidence

The model tests cover:

- all upstream radial basis families, cutoffs, compilation, normalization, and finite gradients;
- scatter values, repeated indices, arbitrary trailing dimensions, empty inputs, compilation, and gradients;
- exact radius-graph edges, batches, rigid-motion invariance, strict boundaries, coincident points, and empty graphs;
- convolution O(3) equivariance over proper rotations and inversion;
- convolution compilation with integer topology, cached reuse, input gradients, and every parameter gradient;
- message invariance to edge ordering and safe zero-edge execution;
- the exact upstream three-layer, 100-neuron model configuration;
- full E(3) equivariance, including translation and inversion;
- batched graphs versus independent evaluation and graph reduction;
- fixed-topology network compilation and reuse;
- gradients through positions, node features, node attributes, and every parameter, followed by a parameter update;
- optional node inputs/attributes, isolated nodes, deepcopy, and MLX weight save/load.

The corresponding tests are:

- `tests/test_graph_radial.py`
- `tests/test_gate_points_2102_convolution.py`
- `tests/test_gate_points_2102_network.py`
