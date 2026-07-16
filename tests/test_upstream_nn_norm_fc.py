"""MLX adaptations of upstream BatchNorm, NormActivation, and FC tests."""

from __future__ import annotations

import pytest

import e3nn_mlx as e3nn
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.compat import require_mlx
from mlx.utils import tree_flatten


def _array(irreps, values):
    return e3nn.IrrepsArray(irreps, values)


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value)))


def _assert_module_equivariant(module, values, tolerance=3e-4) -> None:
    mx = mlx_backend._require()
    angles = e3nn.rand_angles()
    d_in = e3nn.irreps_wigner_d(module.irreps_in, *angles)
    d_out = e3nn.irreps_wigner_d(module.irreps_out, *angles)
    actual = module(_array(module.irreps_in, values @ mx.swapaxes(d_in, -1, -2))).array
    expected = module(_array(module.irreps_in, values)).array @ mx.swapaxes(d_out, -1, -2)
    assert _max_abs(actual - expected) < tolerance


@pytest.mark.mlx
def test_upstream_batchnorm_train_eval_equivariance_and_running_state() -> None:
    mx = mlx_backend._require()
    irreps = e3nn.Irreps("3x0e + 3x0o + 4x1e")
    module = e3nn.BatchNorm(irreps)
    module(_array(irreps, mx.random.normal(shape=(16, irreps.dim))))
    module(_array(irreps, mx.random.normal(shape=(16, irreps.dim))))
    assert module.running_mean.shape == (3,)
    assert module.running_var.shape == (10,)
    module.train()
    _assert_module_equivariant(module, mx.random.normal(shape=(16, irreps.dim)))
    module.eval()
    _assert_module_equivariant(module, mx.random.normal(shape=(16, irreps.dim)))


@pytest.mark.mlx
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("reduce", ["mean", "max"])
@pytest.mark.parametrize("normalization", ["norm", "component"])
@pytest.mark.parametrize("instance", [True, False])
def test_upstream_batchnorm_modes(affine: bool, reduce: str, normalization: str, instance: bool) -> None:
    mx = mlx_backend._require()
    irreps = e3nn.Irreps("10x0e + 5x1e")
    module = e3nn.BatchNorm(
        irreps, affine=affine, reduce=reduce, normalization=normalization, instance=instance
    )
    assert "BatchNorm" in repr(module)
    module.train()
    output = module(_array(irreps, mx.random.normal(shape=(20, 20, irreps.dim))))
    assert output.shape == (20, 20, irreps.dim)
    module.eval()
    output = module(_array(irreps, mx.random.normal(shape=(20, 20, irreps.dim))))
    assert output.shape == (20, 20, irreps.dim)
    leaves = tree_flatten(module.parameters())
    assert bool(leaves) == affine


@pytest.mark.mlx
@pytest.mark.parametrize("instance", [True, False])
@pytest.mark.parametrize("normalization", ["norm", "component"])
def test_upstream_batchnorm_output_statistics(instance: bool, normalization: str) -> None:
    mx = mlx_backend._require()
    batch = samples = 20
    irreps = e3nn.Irreps("3x0e + 4x1e")
    module = e3nn.BatchNorm(irreps, normalization=normalization, instance=instance)
    values = 5.0 * mx.random.normal(shape=(batch, samples, irreps.dim)) + 10.0
    output = module(_array(irreps, values)).array
    scalars = output[..., :3]
    assert _max_abs(mx.mean(scalars, axis=(0, 1))) < 2e-5
    assert _max_abs(mx.mean(scalars**2, axis=(0, 1)) - 1.0) < 2e-4
    vectors = output[..., 3:].reshape(batch, samples, 4, 3)
    statistic = mx.sum(vectors**2, axis=-1) if normalization == "norm" else mx.mean(vectors**2, axis=-1)
    assert _max_abs(mx.mean(statistic, axis=(0, 1)) - 1.0) < 2e-4


@pytest.mark.mlx
@pytest.mark.parametrize("do_bias", [True, False])
@pytest.mark.parametrize("nonlinearity", ["tanh", "sigmoid"])
def test_upstream_norm_activation_values_directions_and_zero_grad(do_bias: bool, nonlinearity: str) -> None:
    mx = mlx_backend._require()
    function = mx.tanh if nonlinearity == "tanh" else mx.sigmoid
    irreps = e3nn.Irreps("4x0e + 5x1o")
    module = e3nn.NormActivation(irreps, function, normalize=True, bias=do_bias)
    if do_bias:
        biases = mx.random.normal(shape=(irreps.num_irreps,))
        module.update({"biases": biases})
        assert set(module.parameters()) == {"biases"}
    else:
        biases = mx.zeros((irreps.num_irreps,))
        assert module.parameters() == {}
    values = mx.random.normal(shape=(3, irreps.dim))
    values = mx.concatenate(
        [
            mx.concatenate([mx.zeros((1, 1)), values[:1, 1:]], axis=-1),
            mx.concatenate([values[1:2, :4], mx.zeros((1, 3)), values[1:2, 7:]], axis=-1),
            values[2:],
        ],
        axis=0,
    )
    output = module(_array(irreps, values)).array
    scalar_input = values[:, :4]
    expected_scalars = mx.sign(scalar_input) * function(mx.abs(scalar_input) + biases[:4])
    assert _max_abs(output[:, :4] - expected_scalars) < 2e-5
    vector_input = values[:, 4:].reshape(3, 5, 3)
    vector_output = output[:, 4:].reshape(3, 5, 3)
    input_norm = mx.sqrt(mx.sum(vector_input**2, axis=-1))
    output_norm = mx.sqrt(mx.sum(vector_output**2, axis=-1))
    expected_norm = mx.abs(function(input_norm + biases[4:]))
    nonzero = input_norm > 1e-7
    assert _max_abs(mx.where(nonzero, output_norm - expected_norm, mx.zeros_like(output_norm))) < 3e-5
    assert _max_abs(mx.where(nonzero, mx.zeros_like(output_norm), output_norm)) == 0.0

    zero = mx.zeros((irreps.dim,))
    gradient = mx.grad(lambda raw: mx.sum(module(_array(irreps, raw)).array))(zero)
    mx.eval(gradient)
    assert bool(mx.all(mx.isfinite(gradient)))


@pytest.mark.mlx
@pytest.mark.parametrize("do_bias", [True, False])
@pytest.mark.parametrize("nonlinearity", ["tanh", "sigmoid"])
def test_upstream_norm_activation_equivariance_and_compile(do_bias: bool, nonlinearity: str) -> None:
    mx = mlx_backend._require()
    function = mx.tanh if nonlinearity == "tanh" else mx.sigmoid
    irreps = e3nn.Irreps("2x0e + 3x0o + 5x1o + 1x1e + 2x2e + 1x2o + 1x3e + 1x3o + 1x5e + 1x6o")
    module = e3nn.NormActivation(irreps, function, bias=do_bias)
    if do_bias:
        module.update({"biases": mx.random.normal(shape=(irreps.num_irreps,))})
    values = mx.random.normal(shape=(5, irreps.dim))
    _assert_module_equivariant(module, values, tolerance=8e-4)
    compiled = mx.compile(lambda raw: module(_array(irreps, raw)).array)
    assert _max_abs(compiled(values) - module(_array(irreps, values)).array) < 2e-6


@pytest.mark.mlx
@pytest.mark.parametrize("act", [None, "tanh"])
@pytest.mark.parametrize(
    "variance_in,variance_out,out_act",
    [(1.0, 1.0, False), (1.0, 1.0, True), (0.1, 10.0, False), (0.1, 0.05, True)],
)
def test_upstream_fully_connected_net_variance_compile_and_gradients(
    act, variance_in: float, variance_out: float, out_act: bool
) -> None:
    mx = mlx_backend._require()
    activation = None if act is None else mx.tanh
    dimensions = (256, 128, 192, 4)
    module = e3nn.FullyConnectedNet(dimensions, activation, variance_in, variance_out, out_act)
    values = mx.random.normal(shape=(4096, dimensions[0])) * variance_in**0.5
    output = module(values) / variance_out**0.5
    if not out_act:
        assert abs(float(mx.mean(output))) < 0.2
    variance = float(mx.mean(output**2))
    assert 0.5 < variance < 2.0
    compiled = mx.compile(module)
    assert _max_abs(compiled(values[:8]) - module(values[:8])) < 3e-5

    loss = lambda model: mx.mean(model(values[:32]) ** 2)
    _, mlx_nn = require_mlx()

    value, gradients = mlx_nn.value_and_grad(module, lambda: loss(module))()
    mx.eval(value, gradients)
    assert len(tree_flatten(gradients)) == len(dimensions) - 1
