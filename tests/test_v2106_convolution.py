"""Extensive tests for the modular v2106 point convolution."""

from __future__ import annotations

import pytest
from mlx.utils import tree_flatten

import e3nn_mlx as e3nn
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.compat import require_mlx
from e3nn_mlx.models.v2106.points_convolution import Convolution


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value))) if value.size else 0.0


def _module() -> Convolution:
    return Convolution(
        "2x0e + 1e",
        "0e + 1e",
        "0e + 1e",
        "2x0e + 1e + 1o",
        [4, 16],
        3.0,
    )


def _inputs(module: Convolution):
    mx = mlx_backend._require()
    positions = mx.array(
        [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [-0.2, 0.8, 0.1], [0.1, -0.3, 0.9]]
    )
    edges = e3nn.radius_graph(positions, 2.0)
    edge_attr = mx.random.normal(shape=(edges.shape[1], module.irreps_edge_attr.dim))
    return (
        mx.random.normal(shape=(4, module.irreps_node_input.dim)),
        mx.random.normal(shape=(4, module.irreps_node_attr.dim)),
        edges[0],
        edges[1],
        edge_attr,
        mx.random.normal(shape=(edges.shape[1], module.fc_neurons[0])),
    )


def _activate_alpha(module: Convolution) -> None:
    mx = mlx_backend._require()
    module.alpha.update({"weight": mx.random.normal(shape=module.alpha.weight.shape)})


@pytest.mark.mlx
def test_v2106_convolution_construction_scalar_path_and_zero_alpha_initialization() -> None:
    mx = mlx_backend._require()
    module = _module()
    values = _inputs(module)
    assert "0e" in module.irreps_mid
    assert _max_abs(module.alpha.weight) == 0.0
    output = module.forward_arrays(*values)
    self_connection = module.sc(
        e3nn.IrrepsArray(module.sc.irreps_in1, values[0]),
        e3nn.IrrepsArray(module.sc.irreps_in2, values[1]),
    ).array
    assert _max_abs(output - self_connection) < 3e-5
    assert output.shape == (4, module.irreps_node_output.dim)
    assert bool(mx.all(module.alpha.output_mask))


@pytest.mark.mlx
def test_v2106_convolution_o3_equivariance_with_active_alpha_branch() -> None:
    mx = mlx_backend._require()
    module = _module()
    _activate_alpha(module)
    values = _inputs(module)
    baseline = module.forward_arrays(*values)
    assert _max_abs(baseline) > 0
    for _ in range(5):
        angles = e3nn.rand_angles()
        for inversion in (0, 1):
            matrices = [
                e3nn.irreps_wigner_d(irreps, *angles, k=inversion)
                for irreps in (
                    module.irreps_node_input,
                    module.irreps_node_attr,
                    module.irreps_edge_attr,
                    module.irreps_node_output,
                )
            ]
            transformed = (
                values[0] @ mx.swapaxes(matrices[0], -1, -2),
                values[1] @ mx.swapaxes(matrices[1], -1, -2),
                values[2],
                values[3],
                values[4] @ mx.swapaxes(matrices[2], -1, -2),
                values[5],
            )
            actual = module.forward_arrays(*transformed)
            expected = baseline @ mx.swapaxes(matrices[3], -1, -2)
            assert _max_abs(actual - expected) < 9e-4


@pytest.mark.mlx
def test_v2106_convolution_compilation_and_cached_reuse() -> None:
    mx = mlx_backend._require()
    module = _module()
    _activate_alpha(module)
    values = _inputs(module)
    expected = module.forward_arrays(*values)
    mx.eval(expected)
    compiled = mx.compile(module.forward_arrays)
    assert _max_abs(compiled(*values) - expected) < 5e-5
    changed = (values[0] * 0.9, values[1], *values[2:])
    changed_expected = module.forward_arrays(*changed)
    mx.eval(changed_expected)
    assert _max_abs(compiled(*changed) - changed_expected) < 5e-5


@pytest.mark.mlx
def test_v2106_convolution_input_and_parameter_gradients() -> None:
    mx, mlx_nn = require_mlx()
    module = _module()
    _activate_alpha(module)
    values = _inputs(module)

    def input_loss(node_input, node_attr, edge_attr, edge_scalars):
        arranged = node_input, node_attr, values[2], values[3], edge_attr, edge_scalars
        return mx.mean(module.forward_arrays(*arranged) ** 2)

    input_gradients = mx.grad(input_loss, argnums=(0, 1, 2, 3))(
        values[0], values[1], values[4], values[5]
    )
    assert all(
        gradient.shape == original.shape and bool(mx.all(mx.isfinite(gradient)))
        for gradient, original in zip(
            input_gradients, (values[0], values[1], values[4], values[5]), strict=True
        )
    )

    value, gradients = mlx_nn.value_and_grad(
        module, lambda: mx.mean(module.forward_arrays(*values) ** 2)
    )()
    mx.eval(value, gradients)
    leaves = tree_flatten(gradients)
    assert len(leaves) == len(tree_flatten(module.trainable_parameters()))
    assert all(bool(mx.all(mx.isfinite(gradient))) for _, gradient in leaves)
    assert any(path.startswith("alpha") and _max_abs(gradient) > 0 for path, gradient in leaves)
    assert any(path.startswith("fc") and _max_abs(gradient) > 0 for path, gradient in leaves)


@pytest.mark.mlx
def test_v2106_convolution_edge_order_and_empty_edges() -> None:
    mx = mlx_backend._require()
    module = _module()
    _activate_alpha(module)
    values = _inputs(module)
    expected = module.forward_arrays(*values)
    permutation = mx.arange(values[2].shape[0] - 1, -1, -1)
    reordered = (
        values[0],
        values[1],
        values[2][permutation],
        values[3][permutation],
        values[4][permutation],
        values[5][permutation],
    )
    assert _max_abs(module.forward_arrays(*reordered) - expected) < 4e-5

    empty = (
        values[0],
        values[1],
        mx.zeros((0,), dtype=mx.int32),
        mx.zeros((0,), dtype=mx.int32),
        mx.zeros((0, module.irreps_edge_attr.dim)),
        mx.zeros((0, module.fc_neurons[0])),
    )
    output = module.forward_arrays(*empty)
    self_connection = module.sc(
        e3nn.IrrepsArray(module.sc.irreps_in1, values[0]),
        e3nn.IrrepsArray(module.sc.irreps_in2, values[1]),
    ).array
    assert _max_abs(output - self_connection) < 3e-5


@pytest.mark.mlx
def test_v2106_convolution_validation() -> None:
    mx = mlx_backend._require()
    module = _module()
    values = list(_inputs(module))
    values[5] = mx.zeros((values[2].shape[0], module.fc_neurons[0] + 1))
    with pytest.raises(ValueError, match="edge_scalars"):
        module.forward_arrays(*values)
    with pytest.raises(ValueError, match="num_neighbors"):
        Convolution("0e", "0e", "0e", "0e", [2, 4], 0.0)
    with pytest.raises(ValueError, match="fc_neurons"):
        Convolution("0e", "0e", "0e", "0e", [], 1.0)
