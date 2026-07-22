"""Extensive tests for the MLX gate-points 2102 convolution."""

from __future__ import annotations

from math import pi, sin

import pytest
from mlx.utils import tree_flatten

import e3nn_mlx as e3nn
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.models.gate_points_2102 import Convolution, tp_path_exists


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value))) if value.size else 0.0


def _convolution() -> Convolution:
    return Convolution(
        "2x0e + 1x1o",
        "3x0e",
        e3nn.Irreps.spherical_harmonics(2),
        "2x0e + 2x0e + 2x1o + 1x1e",
        number_of_edge_features=7,
        radial_layers=2,
        radial_neurons=12,
        num_neighbors=3.0,
    )


def _inputs(module: Convolution):
    mx = mlx_backend._require()
    positions = mx.array(
        [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [-0.2, 0.8, 0.1], [0.1, -0.3, 0.9]]
    )
    edges = e3nn.radius_graph(positions, 2.0)
    edge_vectors = positions[edges[0]] - positions[edges[1]]
    edge_attr = e3nn.spherical_harmonics(
        module.irreps_edge_attr, edge_vectors, normalize=True, normalization="component"
    )
    return (
        mx.random.normal(shape=(positions.shape[0], module.irreps_in.dim)),
        mx.random.normal(shape=(positions.shape[0], module.irreps_node_attr.dim)),
        edges[0],
        edges[1],
        edge_attr,
        mx.random.normal(shape=(edges.shape[1], module.number_of_edge_features)),
    )


def _call(module: Convolution, values):
    node_input, node_attr, edge_src, edge_dst, edge_attr, edge_features = values
    return module(
        e3nn.IrrepsArray(module.irreps_in, node_input),
        e3nn.IrrepsArray(module.irreps_node_attr, node_attr),
        edge_src,
        edge_dst,
        e3nn.IrrepsArray(module.irreps_edge_attr, edge_attr),
        edge_features,
    )


@pytest.mark.mlx
def test_convolution_construction_paths_shape_and_parameter_tree() -> None:
    module = _convolution()
    values = _inputs(module)
    output = _call(module, values)
    assert output.irreps == module.irreps_out
    assert output.shape == (4, module.irreps_out.dim)
    assert module.tp.weight_numel > 0
    assert tp_path_exists(module.irreps_in, module.irreps_edge_attr, "1e")
    assert not tp_path_exists("0e", "0e", "1o")
    parameters = module.parameters()
    assert set(parameters) == {"sc", "lin1", "fc", "tp", "lin2"}
    assert parameters["tp"] == {}
    assert tree_flatten(module.tp.parameters()) == []
    assert all(value.size > 0 for _, value in tree_flatten(parameters))


@pytest.mark.mlx
def test_convolution_o3_equivariance_for_rotations_and_inversion() -> None:
    mx = mlx_backend._require()
    module = _convolution()
    values = _inputs(module)
    baseline = _call(module, values).array
    for _ in range(5):
        angles = e3nn.rand_angles()
        for inversion in (0, 1):
            d_input = e3nn.irreps_wigner_d(module.irreps_in, *angles, k=inversion)
            d_attr = e3nn.irreps_wigner_d(module.irreps_node_attr, *angles, k=inversion)
            d_edge = e3nn.irreps_wigner_d(module.irreps_edge_attr, *angles, k=inversion)
            d_output = e3nn.irreps_wigner_d(module.irreps_out, *angles, k=inversion)
            transformed = (
                values[0] @ mx.swapaxes(d_input, -1, -2),
                values[1] @ mx.swapaxes(d_attr, -1, -2),
                values[2],
                values[3],
                values[4] @ mx.swapaxes(d_edge, -1, -2),
                values[5],
            )
            actual = _call(module, transformed).array
            expected = baseline @ mx.swapaxes(d_output, -1, -2)
            assert _max_abs(actual - expected) < 8e-4


@pytest.mark.mlx
def test_convolution_compiles_with_integer_topology_and_matches_eager() -> None:
    mx = mlx_backend._require()
    module = _convolution()
    values = _inputs(module)
    expected = module.forward_arrays(*values)
    # Materialize the eager graph before compiling the same stateful module.
    # Otherwise both lazy graphs can be evaluated together after compilation,
    # producing a sporadic comparison of differently scheduled scatter sums.
    mx.eval(expected)
    compiled = mx.compile(module.forward_arrays)
    actual = compiled(*values)
    assert _max_abs(actual - expected) < 4e-5

    # A second value set with the same static topology exercises cached code.
    changed = (
        values[0] * 0.7,
        values[1] + 0.2,
        values[2],
        values[3],
        values[4],
        values[5] - 0.1,
    )
    changed_expected = module.forward_arrays(*changed)
    mx.eval(changed_expected)
    assert _max_abs(compiled(*changed) - changed_expected) < 4e-5


@pytest.mark.mlx
def test_convolution_input_and_parameter_gradients_are_finite_and_complete() -> None:
    mx = mlx_backend._require()
    module = _convolution()
    values = _inputs(module)

    def input_loss(node_input, node_attr, edge_attr, edge_features):
        arranged = (node_input, node_attr, values[2], values[3], edge_attr, edge_features)
        return mx.mean(module.forward_arrays(*arranged) ** 2)

    gradients = mx.grad(input_loss, argnums=(0, 1, 2, 3))(
        values[0], values[1], values[4], values[5]
    )
    for gradient, original in zip(gradients, (values[0], values[1], values[4], values[5]), strict=True):
        mx.eval(gradient)
        assert gradient.shape == original.shape
        assert bool(mx.all(mx.isfinite(gradient)))

    from e3nn_mlx.compat import require_mlx

    _, mlx_nn = require_mlx()
    loss = lambda: mx.mean(module.forward_arrays(*values) ** 2)
    value, parameter_gradients = mlx_nn.value_and_grad(module, loss)()
    mx.eval(value, parameter_gradients)
    assert set(parameter_gradients) == {"sc", "lin1", "fc", "tp", "lin2"}
    assert parameter_gradients["tp"] == {}
    leaves = tree_flatten(parameter_gradients)
    assert leaves
    assert all(bool(mx.all(mx.isfinite(gradient))) for _, gradient in leaves)
    assert all(_max_abs(gradient) > 0 for _, gradient in leaves)


@pytest.mark.mlx
def test_convolution_edge_order_invariance_and_empty_neighborhood() -> None:
    mx = mlx_backend._require()
    module = _convolution()
    values = _inputs(module)
    expected = _call(module, values).array
    permutation = mx.arange(values[2].shape[0] - 1, -1, -1)
    reordered = (
        values[0],
        values[1],
        values[2][permutation],
        values[3][permutation],
        values[4][permutation],
        values[5][permutation],
    )
    assert _max_abs(_call(module, reordered).array - expected) < 3e-5

    empty = (
        values[0],
        values[1],
        mx.zeros((0,), dtype=mx.int32),
        mx.zeros((0,), dtype=mx.int32),
        mx.zeros((0, module.irreps_edge_attr.dim)),
        mx.zeros((0, module.number_of_edge_features)),
    )
    output = _call(module, empty).array
    self_connection = module.sc(
        e3nn.IrrepsArray(module.sc.irreps_in1, values[0]),
        e3nn.IrrepsArray(module.sc.irreps_in2, values[1]),
    ).array
    assert _max_abs(output - sin(pi / 8) * self_connection) < 3e-5


@pytest.mark.mlx
def test_convolution_validation() -> None:
    mx = mlx_backend._require()
    module = _convolution()
    values = list(_inputs(module))
    values[5] = mx.zeros((values[2].shape[0], module.number_of_edge_features + 1))
    with pytest.raises(ValueError, match="edge_features"):
        module.forward_arrays(*values)
    values = list(_inputs(module))
    values[2] = values[2].astype(mx.float32)
    with pytest.raises(TypeError, match="integer"):
        module.forward_arrays(*values)
    with pytest.raises(ValueError, match="num_neighbors"):
        Convolution("0e", "0e", "0e", "0e", 2, 1, 4, 0.0)
    with pytest.raises(ValueError, match="no TensorProduct paths"):
        Convolution("0e", "0e", "0e", "1o", 2, 1, 4, 1.0)
