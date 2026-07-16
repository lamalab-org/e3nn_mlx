# Modular v2106 point models

## Scope

This package ports the complete upstream `e3nn.nn.models.v2106` point-model
family to MLX:

- `points_convolution.Convolution`;
- `gate_points_message_passing.MessagePassing` and `Compose`;
- `gate_points_networks.SimpleNetwork`;
- `gate_points_networks.NetworkForAGraphWithAttributes`.

The implementation keeps the upstream tensor-product paths, gate selection,
hidden representation construction, ten-component `smooth_finite` radial
basis, component-normalized spherical harmonics, learned residual mixing, and
`sqrt(num_neighbors)` / `sqrt(num_nodes)` normalizations. Public model calls
return `IrrepsArray` objects so the output representation remains explicit.

## Public API

The canonical imports mirror the upstream module layout:

```python
from e3nn_mlx.models.v2106 import (
    Convolution,
    MessagePassing,
    NetworkForAGraphWithAttributes,
    SimpleNetwork,
)
```

The package root also exports `V2106Convolution`, `V2106MessagePassing`,
`SimpleNetwork`, and `NetworkForAGraphWithAttributes`.

### SimpleNetwork

`SimpleNetwork` derives node attributes, spherical-harmonic edge attributes,
and radial features from node positions:

```python
network = SimpleNetwork(
    irreps_in="3x0e + 2x1o",
    irreps_out="4x0e + 1x1o",
    max_radius=2.0,
    num_neighbors=3.0,
    num_nodes=5.0,
)

output = network({"pos": positions, "x": node_features, "batch": batch})
```

`batch` is optional and defaults to one graph. With `pool_nodes=True`, the
result has one row per graph. With `pool_nodes=False`, it has one row per node.

### NetworkForAGraphWithAttributes

This variant accepts representation-aware node and edge attributes:

```python
network = NetworkForAGraphWithAttributes(
    irreps_node_input="3x0e + 2x1o",
    irreps_node_attr="4x0e + 1x1o",
    irreps_edge_attr="1e",
    irreps_node_output="3x0o + 1e",
    max_radius=2.0,
    num_neighbors=3.0,
    num_nodes=5.0,
)

output = network(
    {
        "pos": positions,
        "node_input": node_features,  # "x" is an equivalent alias
        "node_attr": node_attributes,
        "edge_attr": edge_attributes,
        "edge_index": edge_index,     # optional
        "batch": batch,               # optional
    }
)
```

When `edge_index` is omitted, the model constructs a directed, loop-free
radius graph within each batch. `edge_attr` must follow the resulting edge
order. When attributes come from an existing graph or neighbor list, supplying
`edge_index` explicitly is therefore recommended.

## Fixed-topology compilation

MLX 0.31 has no device-side dynamic `nonzero`, so radius-graph discovery is an
eager operation. Both network classes expose `forward_with_edges` to keep the
entire differentiable calculation inside a compiled MLX graph once a topology
is known.

For `SimpleNetwork`:

```python
edges = e3nn.radius_graph(positions, network.max_radius, batch)

def forward(pos, x, batch, src, dst):
    return network.forward_with_edges(
        pos, x, batch, src, dst, num_graphs=num_graphs
    ).array

compiled_forward = mx.compile(forward)
```

For `NetworkForAGraphWithAttributes`, pass `node_attr` and the caller-provided
`edge_attr` between `node_input` and `batch`. Position gradients continue
through edge vectors, distances, the radial basis, and spherical harmonics;
only the discrete edge selection is fixed.

## Lower-level modules

`Convolution` accepts `IrrepsArray` node features, node attributes, and edge
attributes plus integer `edge_src` / `edge_dst` arrays and invariant radial
features. `forward_arrays` is the raw-array compilation entry point.

The v2106 residual rule is preserved exactly. The learned scalar tensor product
`alpha` starts at zero, so paths supported by the self connection initially
equal that self connection. Unlike a conventional fixed residual coefficient,
`alpha` becomes an invariant, data-dependent mixing factor during training.

`MessagePassing` builds a gated convolution for every hidden representation,
filters unreachable tensor-product paths, tracks the actual post-gate irrep
sequence, and finishes with an ungated convolution. It also provides
`forward_arrays`.

Empty edge lists are supported by every layer. They bypass the per-edge tensor
product and produce the same residual-only result without invoking unsupported
zero-sized compiled Metal kernels.

## Verification evidence

The focused v2106 tests cover:

- scalar middle paths, zero-initialized and active learned residual mixing;
- repeated proper rotations and inversion for convolution, message passing,
  and both complete networks;
- translation invariance of the complete networks;
- exact upstream-sized three-hidden-layer configurations;
- compiled execution and cached reuse at every layer of the public stack;
- gradients for positions, node inputs, node attributes, edge attributes,
  radial inputs, and every trainable parameter;
- an optimizer-style parameter update;
- edge-order invariance, empty edges, isolated nodes, and automatic graphs;
- batched graphs versus independent evaluation and pooled versus node output;
- `x` / `node_input` aliases, validation failures, deepcopy, and MLX weight
  save/load round-trips.

The corresponding tests are:

- `tests/test_v2106_convolution.py`;
- `tests/test_v2106_message_passing.py`;
- `tests/test_v2106_networks.py`.
