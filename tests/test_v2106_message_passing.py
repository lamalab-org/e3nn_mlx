"""Extensive tests for modular v2106 gated message passing."""

from __future__ import annotations

import copy

import pytest

tree_flatten = pytest.importorskip("mlx.utils").tree_flatten

import e3nn_mlx as e3nn
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.compat import require_mlx
from e3nn_mlx.models.v2106 import Compose, MessagePassing, tp_path_exists


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value))) if value.size else 0.0


def _module() -> MessagePassing:
    return MessagePassing(
        ["0e", "0e + 1e", "0e + 1e", "0e + 1e", "1e"],
        "0e + 1e",
        "0e + 1e",
        [2, 16],
        3.0,
    )


def _inputs(module: MessagePassing):
    mx = mlx_backend._require()
    positions = mx.array(
        [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [-0.2, 0.8, 0.1], [0.1, -0.3, 0.9]]
    )
    edges = e3nn.radius_graph(positions, 2.0)
    return (
        mx.random.normal(shape=(4, module.irreps_node_input.dim)),
        mx.random.normal(shape=(4, module.irreps_node_attr.dim)),
        edges[0],
        edges[1],
        mx.random.normal(shape=(edges.shape[1], module.irreps_edge_attr.dim)),
        mx.random.normal(shape=(edges.shape[1], module.fc_neurons[0])),
    )


def _activate_alpha(module: MessagePassing) -> None:
    mx = mlx_backend._require()
    for layer in module.layers:
        convolution = layer.first if isinstance(layer, Compose) else layer
        convolution.alpha.update(
            {"weight": 0.05 * mx.random.normal(shape=convolution.alpha.weight.shape)}
        )


@pytest.mark.mlx
def test_v2106_message_passing_upstream_configuration_and_path_filtering() -> None:
    module = _module()
    output = module.forward_arrays(*_inputs(module))
    assert output.shape == (4, 3)
    assert len(module.layers) == 4
    assert all(isinstance(layer, Compose) for layer in module.layers[:-1])
    assert module.irreps_node_input == e3nn.Irreps("0e")
    assert module.irreps_node_output == e3nn.Irreps("1e")
    assert len(module.irreps_node_sequence) == 5
    assert tp_path_exists("1e", "1e", "0e")
    assert not tp_path_exists("0e", "0e", "1e")


@pytest.mark.mlx
@pytest.mark.parametrize("enabled", [False, True])
def test_v2106_message_passing_propagates_kernel_mode_and_preserves_it_on_copy(
    enabled,
) -> None:
    module = MessagePassing(
        ["0e", "0e + 1e", "1e"],
        "0e",
        "0e + 1e",
        [2, 8],
        3.0,
        use_custom_kernel=enabled,
    )
    convolutions = [
        layer.first if isinstance(layer, Compose) else layer for layer in module.layers
    ]
    assert module.use_custom_kernel is enabled
    assert all(layer.use_custom_kernel is enabled for layer in convolutions)
    assert copy.deepcopy(module).use_custom_kernel is enabled


@pytest.mark.mlx
def test_v2106_message_passing_o3_equivariance_with_all_alpha_branches_active() -> None:
    mx = mlx_backend._require()
    module = _module()
    _activate_alpha(module)
    values = _inputs(module)
    baseline = module.forward_arrays(*values)
    for _ in range(5):
        angles = e3nn.rand_angles()
        for inversion in (0, 1):
            d_input = e3nn.irreps_wigner_d(module.irreps_node_input, *angles, k=inversion)
            d_node_attr = e3nn.irreps_wigner_d(
                module.irreps_node_attr, *angles, k=inversion
            )
            d_edge_attr = e3nn.irreps_wigner_d(
                module.irreps_edge_attr, *angles, k=inversion
            )
            d_output = e3nn.irreps_wigner_d(module.irreps_node_output, *angles, k=inversion)
            transformed = (
                values[0] @ mx.swapaxes(d_input, -1, -2),
                values[1] @ mx.swapaxes(d_node_attr, -1, -2),
                values[2],
                values[3],
                values[4] @ mx.swapaxes(d_edge_attr, -1, -2),
                values[5],
            )
            actual = module.forward_arrays(*transformed)
            expected = baseline @ mx.swapaxes(d_output, -1, -2)
            assert _max_abs(actual - expected) < 2e-3


@pytest.mark.mlx
def test_v2106_message_passing_compilation_and_cached_reuse() -> None:
    mx = mlx_backend._require()
    module = _module()
    _activate_alpha(module)
    values = _inputs(module)
    expected = module.forward_arrays(*values)
    mx.eval(expected)
    compiled = mx.compile(module.forward_arrays)
    assert _max_abs(compiled(*values) - expected) < 5e-4
    changed = (values[0] * 1.1, values[1], *values[2:])
    changed_expected = module.forward_arrays(*changed)
    mx.eval(changed_expected)
    assert _max_abs(compiled(*changed) - changed_expected) < 5e-4


@pytest.mark.mlx
def test_v2106_message_passing_input_and_parameter_gradients() -> None:
    mx, mlx_nn = require_mlx()
    module = _module()
    _activate_alpha(module)
    values = _inputs(module)

    def input_loss(node_input, node_attr, edge_attr, edge_scalars):
        arranged = node_input, node_attr, values[2], values[3], edge_attr, edge_scalars
        return mx.mean(module.forward_arrays(*arranged) ** 2)

    gradients = mx.grad(input_loss, argnums=(0, 1, 2, 3))(
        values[0], values[1], values[4], values[5]
    )
    for gradient, original in zip(
        gradients, (values[0], values[1], values[4], values[5]), strict=True
    ):
        assert gradient.shape == original.shape
        assert bool(mx.all(mx.isfinite(gradient)))

    value, parameter_gradients = mlx_nn.value_and_grad(
        module, lambda: mx.mean(module.forward_arrays(*values) ** 2)
    )()
    mx.eval(value, parameter_gradients)
    leaves = tree_flatten(parameter_gradients)
    assert len(leaves) == len(tree_flatten(module.trainable_parameters()))
    assert all(bool(mx.all(mx.isfinite(gradient))) for _, gradient in leaves)
    assert any("alpha" in path and _max_abs(gradient) > 0 for path, gradient in leaves)
    assert any("fc" in path and _max_abs(gradient) > 0 for path, gradient in leaves)


@pytest.mark.mlx
def test_v2106_message_passing_empty_edges_and_deepcopy() -> None:
    mx = mlx_backend._require()
    module = _module()
    _activate_alpha(module)
    values = _inputs(module)
    empty = (
        values[0],
        values[1],
        mx.zeros((0,), dtype=mx.int32),
        mx.zeros((0,), dtype=mx.int32),
        mx.zeros((0, module.irreps_edge_attr.dim)),
        mx.zeros((0, module.fc_neurons[0])),
    )
    output = module.forward_arrays(*empty)
    assert output.shape == (4, module.irreps_node_output.dim)
    assert bool(mx.all(mx.isfinite(output)))

    copied = copy.deepcopy(module)
    assert _max_abs(copied.forward_arrays(*values) - module.forward_arrays(*values)) < 5e-6


@pytest.mark.mlx
def test_v2106_message_passing_validation() -> None:
    with pytest.raises(ValueError, match="input and output"):
        MessagePassing(["0e"], "0e", "0e", [2, 4], 1.0)
    with pytest.raises(ValueError, match="fc_neurons"):
        MessagePassing(["0e", "0e"], "0e", "0e", [], 1.0)
    with pytest.raises(ValueError, match="num_neighbors"):
        MessagePassing(["0e", "0e"], "0e", "0e", [2, 4], 0.0)
