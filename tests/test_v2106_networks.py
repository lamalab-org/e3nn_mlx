"""Extensive end-to-end tests for the modular v2106 point networks."""

from __future__ import annotations

import copy

import pytest

mlx_utils = pytest.importorskip("mlx.utils")
tree_flatten = mlx_utils.tree_flatten
tree_map = mlx_utils.tree_map

import e3nn_mlx as e3nn
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.compat import require_mlx
from e3nn_mlx.models.v2106 import (
    Compose,
    NetworkForAGraphWithAttributes,
    SimpleNetwork,
)


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value))) if value.size else 0.0


def _activate_alpha(module) -> None:
    mx = mlx_backend._require()
    for layer in module.mp.layers:
        convolution = layer.first if isinstance(layer, Compose) else layer
        convolution.alpha.update(
            {"weight": 0.05 * mx.random.normal(shape=convolution.alpha.weight.shape)}
        )


def _positions():
    mx = mlx_backend._require()
    return mx.array(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.1, -0.1],
            [-0.3, 0.7, 0.2],
            [0.2, -0.4, 0.8],
            [-0.5, -0.2, -0.6],
        ]
    )


def _simple(*, exact=False, pool_nodes=True):
    return SimpleNetwork(
        "3x0e + 2x1o",
        "4x0e + 1x1o",
        max_radius=2.0,
        num_neighbors=3.0,
        num_nodes=5.0,
        mul=50 if exact else 4,
        layers=3 if exact else 1,
        lmax=2,
        pool_nodes=pool_nodes,
    )


def _simple_data(module, positions=None):
    mx = mlx_backend._require()
    positions = _positions() if positions is None else positions
    return {
        "pos": positions,
        "x": mx.random.normal(shape=(positions.shape[0], module.irreps_in.dim)),
    }


def _attributed(*, exact=False, pool_nodes=True):
    return NetworkForAGraphWithAttributes(
        "3x0e + 2x1o",
        "4x0e + 1x1o",
        "1e",
        "3x0o + 1e",
        max_radius=2.0,
        num_neighbors=3.0,
        num_nodes=5.0,
        mul=50 if exact else 4,
        layers=3 if exact else 1,
        lmax=2,
        pool_nodes=pool_nodes,
    )


def _attributed_data(module, *, automatic_edges=False):
    mx = mlx_backend._require()
    positions = _positions()
    batch = mx.zeros((positions.shape[0],), dtype=mx.int32)
    edges = e3nn.radius_graph(positions, module.max_radius, batch)
    data = {
        "pos": positions,
        "node_input": mx.random.normal(
            shape=(positions.shape[0], module.irreps_node_input.dim)
        ),
        "node_attr": mx.random.normal(
            shape=(positions.shape[0], module.irreps_node_attr.dim)
        ),
        "edge_attr": mx.random.normal(
            shape=(edges.shape[1], module.irreps_edge_attr.dim)
        ),
    }
    if not automatic_edges:
        data["edge_index"] = edges
    return data


def _spatial_transform(positions, angles, inversion, translation):
    mx = mlx_backend._require()
    rotation = e3nn.rotation_matrix(*angles)
    spatial = ((-1.0) ** inversion) * rotation
    return positions @ mx.swapaxes(spatial, -1, -2) + translation


@pytest.mark.mlx
def test_v2106_exact_upstream_network_configurations_construct_and_run() -> None:
    simple = _simple(exact=True)
    simple_data = _simple_data(simple)
    simple_output = simple(simple_data)
    assert simple_output.irreps == simple.irreps_out
    assert simple_output.shape == (1, simple.irreps_out.dim)
    assert len(simple.mp.layers) == 4
    assert simple.number_of_basis == 10
    assert len(simple.preprocess(simple_data)) == 5

    attributed = _attributed(exact=True)
    attributed_data = _attributed_data(attributed)
    attributed_output = attributed(attributed_data)
    assert attributed_output.irreps == attributed.irreps_node_output
    assert attributed_output.shape == (1, attributed.irreps_node_output.dim)
    assert attributed.mp.irreps_edge_attr == (
        attributed.irreps_edge_attr + attributed.irreps_sh
    )
    assert len(attributed.preprocess(attributed_data)) == 7
    assert e3nn.SimpleNetwork is SimpleNetwork
    assert e3nn.NetworkForAGraphWithAttributes is NetworkForAGraphWithAttributes
    assert tree_flatten(simple.parameters())
    assert tree_flatten(attributed.parameters())


@pytest.mark.mlx
@pytest.mark.parametrize("kind", ["simple", "attributed"])
def test_v2106_network_e3_equivariance_rotation_inversion_translation(kind) -> None:
    mx = mlx_backend._require()
    module = _simple() if kind == "simple" else _attributed()
    data = _simple_data(module) if kind == "simple" else _attributed_data(module)
    _activate_alpha(module)
    baseline = module(data).array
    for _ in range(3):
        angles = e3nn.rand_angles()
        translation = mx.random.normal(shape=(3,)) * 4.0
        for inversion in (0, 1):
            transformed = dict(data)
            transformed["pos"] = _spatial_transform(
                data["pos"], angles, inversion, translation
            )
            d_input = e3nn.irreps_wigner_d(
                module.irreps_node_input, *angles, k=inversion
            )
            input_key = "x" if kind == "simple" else "node_input"
            transformed[input_key] = data[input_key] @ mx.swapaxes(d_input, -1, -2)
            if kind == "attributed":
                d_node_attr = e3nn.irreps_wigner_d(
                    module.irreps_node_attr, *angles, k=inversion
                )
                d_edge_attr = e3nn.irreps_wigner_d(
                    module.irreps_edge_attr, *angles, k=inversion
                )
                transformed["node_attr"] = data["node_attr"] @ mx.swapaxes(
                    d_node_attr, -1, -2
                )
                transformed["edge_attr"] = data["edge_attr"] @ mx.swapaxes(
                    d_edge_attr, -1, -2
                )
            actual = module(transformed).array
            d_output = e3nn.irreps_wigner_d(
                module.irreps_node_output, *angles, k=inversion
            )
            expected = baseline @ mx.swapaxes(d_output, -1, -2)
            assert _max_abs(actual - expected) < 4e-3


@pytest.mark.mlx
def test_v2106_simple_network_batched_graphs_and_pooling_match_independent_runs() -> None:
    mx = mlx_backend._require()
    module = _simple()
    first = _simple_data(module, mx.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]))
    second = _simple_data(
        module,
        mx.array([[0.0, 0.0, 0.0], [-0.6, 0.2, 0.0], [0.1, 0.7, 0.0]]),
    )
    independent = mx.concatenate([module(first).array, module(second).array], axis=0)
    combined = {
        "pos": mx.concatenate([first["pos"], second["pos"]], axis=0),
        "x": mx.concatenate([first["x"], second["x"]], axis=0),
        "batch": mx.array([0, 0, 1, 1, 1], dtype=mx.int32),
    }
    pooled = module(combined).array
    assert _max_abs(pooled - independent) < 8e-5

    module.pool_nodes = False
    node_output = module(combined).array
    expected = e3nn.scatter_sum(node_output, combined["batch"], 2) / (module.num_nodes**0.5)
    assert _max_abs(expected - pooled) < 5e-5


@pytest.mark.mlx
@pytest.mark.parametrize("kind", ["simple", "attributed"])
def test_v2106_network_fixed_topology_compiles_and_reuses(kind) -> None:
    mx = mlx_backend._require()
    module = _simple() if kind == "simple" else _attributed()
    data = _simple_data(module) if kind == "simple" else _attributed_data(module)
    batch = mx.zeros((data["pos"].shape[0],), dtype=mx.int32)
    edges = e3nn.radius_graph(data["pos"], module.max_radius, batch)

    if kind == "simple":
        def forward(pos, node_input, graph_batch, edge_src, edge_dst):
            return module.forward_with_edges(
                pos, node_input, graph_batch, edge_src, edge_dst, num_graphs=1
            ).array

        args = data["pos"], data["x"], batch, edges[0], edges[1]
    else:
        def forward(pos, node_input, node_attr, edge_attr, graph_batch, edge_src, edge_dst):
            return module.forward_with_edges(
                pos,
                node_input,
                node_attr,
                edge_attr,
                graph_batch,
                edge_src,
                edge_dst,
                num_graphs=1,
            ).array

        args = (
            data["pos"],
            data["node_input"],
            data["node_attr"],
            data["edge_attr"],
            batch,
            edges[0],
            edges[1],
        )
    expected = forward(*args)
    mx.eval(expected)
    compiled = mx.compile(forward)
    assert _max_abs(compiled(*args) - expected) < 2e-4
    changed = list(args)
    changed[1] = changed[1] * 0.9
    changed_expected = forward(*changed)
    mx.eval(changed_expected)
    assert _max_abs(compiled(*changed) - changed_expected) < 2e-4


@pytest.mark.mlx
@pytest.mark.parametrize("kind", ["simple", "attributed"])
def test_v2106_network_all_input_and_parameter_gradients_and_training(kind) -> None:
    mx, mlx_nn = require_mlx()
    module = _simple() if kind == "simple" else _attributed()
    data = _simple_data(module) if kind == "simple" else _attributed_data(module)
    _activate_alpha(module)
    batch = mx.zeros((data["pos"].shape[0],), dtype=mx.int32)
    edges = e3nn.radius_graph(data["pos"], module.max_radius, batch)

    if kind == "simple":
        def raw_loss(pos, node_input):
            output = module.forward_with_edges(
                pos, node_input, batch, edges[0], edges[1], num_graphs=1
            ).array
            return mx.mean(output**2)

        originals = data["pos"], data["x"]
    else:
        def raw_loss(pos, node_input, node_attr, edge_attr):
            output = module.forward_with_edges(
                pos,
                node_input,
                node_attr,
                edge_attr,
                batch,
                edges[0],
                edges[1],
                num_graphs=1,
            ).array
            return mx.mean(output**2)

        originals = data["pos"], data["node_input"], data["node_attr"], data["edge_attr"]
    gradients = mx.grad(raw_loss, argnums=tuple(range(len(originals))))(*originals)
    for gradient, original in zip(gradients, originals, strict=True):
        mx.eval(gradient)
        assert gradient.shape == original.shape
        assert bool(mx.all(mx.isfinite(gradient)))

    value, parameter_gradients = mlx_nn.value_and_grad(
        module, lambda: raw_loss(*originals)
    )()
    mx.eval(value, parameter_gradients)
    leaves = tree_flatten(parameter_gradients)
    assert len(leaves) == len(tree_flatten(module.trainable_parameters()))
    assert all(bool(mx.all(mx.isfinite(gradient))) for _, gradient in leaves)
    assert any(_max_abs(gradient) > 1e-8 for _, gradient in leaves)

    before = raw_loss(*originals)
    module.update(
        tree_map(
            lambda parameter, gradient: parameter - 1e-2 * gradient,
            module.parameters(),
            parameter_gradients,
        )
    )
    after = raw_loss(*originals)
    assert abs(float(after - before)) > 1e-9


@pytest.mark.mlx
def test_v2106_network_automatic_edges_alias_empty_edges_and_nonpooled_output() -> None:
    mx = mlx_backend._require()
    module = _attributed(pool_nodes=False)
    automatic = _attributed_data(module, automatic_edges=True)
    automatic["x"] = automatic.pop("node_input")
    output = module(automatic)
    assert output.shape == (5, module.irreps_node_output.dim)

    isolated_positions = mx.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
    )
    isolated = {
        "pos": isolated_positions,
        "x": mx.random.normal(shape=(3, module.irreps_node_input.dim)),
        "node_attr": mx.random.normal(shape=(3, module.irreps_node_attr.dim)),
        "edge_attr": mx.zeros((0, module.irreps_edge_attr.dim)),
    }
    result = module(isolated)
    assert result.shape == (3, module.irreps_node_output.dim)
    assert bool(mx.all(mx.isfinite(result.array)))


@pytest.mark.mlx
@pytest.mark.parametrize("kind", ["simple", "attributed"])
def test_v2106_network_deepcopy_and_weight_round_trip(kind, tmp_path) -> None:
    module = _simple() if kind == "simple" else _attributed()
    data = _simple_data(module) if kind == "simple" else _attributed_data(module)
    _activate_alpha(module)
    expected = module(data).array
    copied = copy.deepcopy(module)
    assert _max_abs(copied(data).array - expected) < 1e-5

    path = tmp_path / f"v2106_{kind}.npz"
    module.save_weights(str(path))
    restored = _simple() if kind == "simple" else _attributed()
    restored.load_weights(str(path))
    assert _max_abs(restored(data).array - expected) < 1e-5


@pytest.mark.mlx
def test_v2106_network_validation() -> None:
    mx = mlx_backend._require()
    simple = _simple()
    with pytest.raises(KeyError, match="pos"):
        simple({})
    with pytest.raises(KeyError, match="x"):
        simple({"pos": mx.zeros((2, 3))})
    with pytest.raises(ValueError, match="same nodes"):
        simple.forward_with_edges(
            mx.zeros((2, 3)),
            mx.zeros((1, simple.irreps_in.dim)),
            mx.zeros((2,), dtype=mx.int32),
            mx.zeros((0,), dtype=mx.int32),
            mx.zeros((0,), dtype=mx.int32),
            num_graphs=1,
        )
    attributed = _attributed()
    with pytest.raises(KeyError, match="node_attr"):
        attributed({"pos": mx.zeros((2, 3)), "x": mx.zeros((2, attributed.irreps_in.dim))})
    with pytest.raises(ValueError, match="edge_index"):
        data = _attributed_data(attributed)
        data["edge_index"] = mx.zeros((3, 2), dtype=mx.int32)
        attributed(data)
    with pytest.raises(ValueError, match="positive"):
        SimpleNetwork("0e", "0e", 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        NetworkForAGraphWithAttributes("0e", "0e", "0e", "0e", 1, 1, 1, layers=-1)
