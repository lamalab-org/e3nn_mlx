"""Extensive end-to-end tests for the MLX gate-points 2102 network."""

from __future__ import annotations

import copy

import pytest

mlx_utils = pytest.importorskip("mlx.utils")
tree_flatten = mlx_utils.tree_flatten
tree_map = mlx_utils.tree_map

import e3nn_mlx as e3nn
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.compat import require_mlx
from e3nn_mlx.models.gate_points_2102 import Network


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value))) if value.size else 0.0


def _network(*, exact: bool = False, reduce_output: bool = True, optional_inputs: bool = False):
    return Network(
        None if optional_inputs else "3x0e + 2x1o",
        "5x0e + 5x0o + 5x1e + 5x1o",
        "2x0o + 2x1o + 2x2e",
        None if optional_inputs else "10x0e",
        e3nn.Irreps.spherical_harmonics(3),
        layers=3 if exact else 1,
        max_radius=2.0,
        number_of_basis=5,
        radial_layers=2 if exact else 1,
        radial_neurons=100 if exact else 12,
        num_neighbors=4.0,
        num_nodes=5.0,
        reduce_output=reduce_output,
    )


def _graph(module: Network, positions=None):
    mx = mlx_backend._require()
    if positions is None:
        positions = mx.array(
            [
                [0.0, 0.0, 0.0],
                [0.6, 0.1, -0.1],
                [-0.3, 0.7, 0.2],
                [0.2, -0.4, 0.8],
                [-0.5, -0.2, -0.6],
            ]
        )
    data = {"pos": positions}
    if module.irreps_in is not None:
        data["x"] = mx.random.normal(shape=(positions.shape[0], module.irreps_in.dim))
    if module.input_has_node_attr:
        data["z"] = mx.random.normal(
            shape=(positions.shape[0], module.irreps_node_attr.dim)
        )
    return data


def _transform_data(module: Network, data, angles, inversion: int, translation=None):
    mx = mlx_backend._require()
    rotation = e3nn.rotation_matrix(*angles)
    spatial = ((-1.0) ** inversion) * rotation
    transformed = dict(data)
    transformed["pos"] = data["pos"] @ mx.swapaxes(spatial, -1, -2)
    if translation is not None:
        transformed["pos"] = transformed["pos"] + translation
    if module.irreps_in is not None:
        d_input = e3nn.irreps_wigner_d(module.irreps_in, *angles, k=inversion)
        transformed["x"] = data["x"] @ mx.swapaxes(d_input, -1, -2)
    d_attr = e3nn.irreps_wigner_d(module.irreps_node_attr, *angles, k=inversion)
    transformed["z"] = data["z"] @ mx.swapaxes(d_attr, -1, -2)
    return transformed


@pytest.mark.mlx
def test_exact_upstream_network_constructs_and_runs_expected_shape() -> None:
    module = _network(exact=True)
    data = _graph(module)
    output = module(data)
    assert output.irreps == module.irreps_out
    assert output.shape == (1, module.irreps_out.dim)
    assert len(module.layers) == 4
    assert "Network" in repr(module)
    parameters = tree_flatten(module.parameters())
    assert parameters
    assert all(value.size > 0 for _, value in parameters)


@pytest.mark.mlx
def test_exact_upstream_network_e3_equivariance_rotation_inversion_translation() -> None:
    mx = mlx_backend._require()
    module = _network(exact=True)
    data = _graph(module)
    baseline = module(data).array
    for _ in range(3):
        angles = e3nn.rand_angles()
        translation = mx.random.normal(shape=(3,)) * 5.0
        for inversion in (0, 1):
            transformed = _transform_data(module, data, angles, inversion, translation)
            actual = module(transformed).array
            d_output = e3nn.irreps_wigner_d(module.irreps_out, *angles, k=inversion)
            expected = baseline @ mx.swapaxes(d_output, -1, -2)
            assert _max_abs(actual - expected) < 3e-3


@pytest.mark.mlx
def test_network_batched_graphs_equal_independent_evaluation_and_node_reduction() -> None:
    mx = mlx_backend._require()
    module = _network()
    first = _graph(module, mx.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.8, 0.0]]))
    second = _graph(
        module,
        mx.array([[0.0, 0.0, 0.0], [-0.6, 0.2, 0.0], [0.1, 0.7, 0.0], [0.0, 0.0, 0.9]]),
    )
    independent = mx.concatenate([module(first).array, module(second).array], axis=0)
    combined = {
        "pos": mx.concatenate([first["pos"], second["pos"]], axis=0),
        "x": mx.concatenate([first["x"], second["x"]], axis=0),
        "z": mx.concatenate([first["z"], second["z"]], axis=0),
        "batch": mx.array([0, 0, 0, 1, 1, 1, 1], dtype=mx.int32),
    }
    batched = module(combined).array
    assert _max_abs(batched - independent) < 5e-5

    module.reduce_output = False
    node_output = module(combined).array
    reduced = e3nn.scatter_sum(node_output, combined["batch"], 2) / (module.num_nodes**0.5)
    assert _max_abs(reduced - batched) < 3e-5


@pytest.mark.mlx
def test_network_fixed_topology_compiles_and_reuses_compiled_graph() -> None:
    mx = mlx_backend._require()
    module = _network()
    data = _graph(module)
    batch = mx.zeros((data["pos"].shape[0],), dtype=mx.int32)
    edges = e3nn.radius_graph(data["pos"], module.max_radius, batch)

    def forward(pos, node_input, node_attr, graph_batch, edge_src, edge_dst):
        return module.forward_with_edges(
            pos,
            node_input,
            node_attr,
            graph_batch,
            edge_src,
            edge_dst,
            num_graphs=1,
        ).array

    expected = forward(data["pos"], data["x"], data["z"], batch, edges[0], edges[1])
    mx.eval(expected)
    compiled = mx.compile(forward)
    actual = compiled(data["pos"], data["x"], data["z"], batch, edges[0], edges[1])
    assert _max_abs(actual - expected) < 8e-5
    changed_x = data["x"] * 0.8
    changed_expected = forward(
        data["pos"], changed_x, data["z"], batch, edges[0], edges[1]
    )
    mx.eval(changed_expected)
    assert _max_abs(
        compiled(data["pos"], changed_x, data["z"], batch, edges[0], edges[1])
        - changed_expected
    ) < 8e-5


@pytest.mark.mlx
def test_network_position_feature_and_all_parameter_gradients_and_training_step() -> None:
    mx, mlx_nn = require_mlx()
    module = _network()
    data = _graph(module)
    batch = mx.zeros((data["pos"].shape[0],), dtype=mx.int32)
    edges = e3nn.radius_graph(data["pos"], module.max_radius, batch)

    def raw_loss(pos, node_input, node_attr):
        output = module.forward_with_edges(
            pos, node_input, node_attr, batch, edges[0], edges[1], num_graphs=1
        ).array
        return mx.mean(output**2)

    input_gradients = mx.grad(raw_loss, argnums=(0, 1, 2))(
        data["pos"], data["x"], data["z"]
    )
    for gradient, original in zip(
        input_gradients, (data["pos"], data["x"], data["z"]), strict=True
    ):
        mx.eval(gradient)
        assert gradient.shape == original.shape
        assert bool(mx.all(mx.isfinite(gradient)))

    loss = lambda: raw_loss(data["pos"], data["x"], data["z"])
    value, gradients = mlx_nn.value_and_grad(module, loss)()
    mx.eval(value, gradients)
    leaves = tree_flatten(gradients)
    assert len(leaves) == len(tree_flatten(module.trainable_parameters()))
    assert leaves
    assert all(bool(mx.all(mx.isfinite(gradient))) for _, gradient in leaves)
    assert any(_max_abs(gradient) > 1e-8 for _, gradient in leaves)

    before = module.forward_with_edges(
        data["pos"], data["x"], data["z"], batch, edges[0], edges[1], num_graphs=1
    ).array
    updated = tree_map(lambda parameter, gradient: parameter - 1e-2 * gradient, module.parameters(), gradients)
    module.update(updated)
    after = module.forward_with_edges(
        data["pos"], data["x"], data["z"], batch, edges[0], edges[1], num_graphs=1
    ).array
    assert _max_abs(after - before) > 1e-7


@pytest.mark.mlx
def test_network_deepcopy_and_weight_round_trip(tmp_path) -> None:
    module = _network(exact=True)
    data = _graph(module)
    expected = module(data).array
    copied = copy.deepcopy(module)
    assert _max_abs(copied(data).array - expected) < 1e-5

    path = tmp_path / "gate_points_2102_weights.npz"
    module.save_weights(str(path))
    restored = _network(exact=True)
    restored.load_weights(str(path))
    assert _max_abs(restored(data).array - expected) < 1e-5


@pytest.mark.mlx
def test_network_optional_inputs_isolated_nodes_and_nonreduced_output() -> None:
    mx = mlx_backend._require()
    module = _network(optional_inputs=True, reduce_output=False)
    positions = mx.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    output = module({"pos": positions})
    assert output.shape == (3, module.irreps_out.dim)
    assert bool(mx.all(mx.isfinite(output.array)))


@pytest.mark.mlx
def test_network_input_validation() -> None:
    mx = mlx_backend._require()
    module = _network()
    with pytest.raises(KeyError, match="pos"):
        module({})
    with pytest.raises(KeyError, match="x"):
        module({"pos": mx.zeros((2, 3)), "z": mx.zeros((2, 10))})
    with pytest.raises(KeyError, match="z"):
        module({"pos": mx.zeros((2, 3)), "x": mx.zeros((2, module.irreps_in.dim))})
    data = _graph(module)
    batch = mx.zeros((data["pos"].shape[0],), dtype=mx.int32)
    edges = e3nn.radius_graph(data["pos"], module.max_radius, batch)
    with pytest.raises(ValueError, match="same nodes"):
        module.forward_with_edges(
            data["pos"], data["x"][:-1], data["z"], batch, edges[0], edges[1], num_graphs=1
        )
    with pytest.raises(ValueError, match="positive"):
        _network().__class__(
            "0e", "0e", "0e", "0e", "0e", 1, 0.0, 2, 1, 4, 1.0, 1.0
        )
