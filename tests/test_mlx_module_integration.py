from __future__ import annotations

import pytest

from e3nn_mlx import FullyConnectedTensorProduct, Gate, IrrepsArray, Linear, Norm
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.compat import require_mlx
from e3nn_mlx.ops_rotations import irreps_wigner_d


pytestmark = pytest.mark.mlx


def _rotate(array: IrrepsArray, alpha, beta, gamma) -> IrrepsArray:
    mx, _ = require_mlx()
    matrix = irreps_wigner_d(array.irreps, alpha, beta, gamma)
    return IrrepsArray(array.irreps, array.array @ mx.swapaxes(matrix, -1, -2))


def _make_model(*, compile_layers: bool):
    _, nn = require_mlx()

    class InvariantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = Linear("1o", "2x1o", bias=False, compile=compile_layers)
            self.product = FullyConnectedTensorProduct(
                "2x1o", "1o", "2x0e+2x1e+2x2e", compile_left_right=compile_layers
            )
            self.output = Linear("2x0e+2x1e+2x2e", "0e", bias=True, compile=compile_layers)

        def __call__(self, vector: IrrepsArray, edge: IrrepsArray) -> IrrepsArray:
            return self.output(self.product(self.input(vector), edge))

    return InvariantModel()


def _training_data():
    vectors = IrrepsArray(
        "1o",
        mlx_backend.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, -0.5, 0.7], [-0.4, 0.1, 0.8]]
        ),
    )
    edges = IrrepsArray(
        "1o",
        mlx_backend.asarray(
            [[0.5, 0.2, -0.1], [-0.3, 0.7, 0.2], [0.6, 0.1, -0.2], [0.2, -0.4, 0.9]]
        ),
    )
    targets = mlx_backend.asarray([[0.5], [0.7], [-0.07], [0.6]])
    return vectors, edges, targets


def test_nested_model_discovers_only_trainable_leaf_parameters() -> None:
    model = _make_model(compile_layers=False)
    assert set(model.parameters()) == {"input", "product", "output"}
    assert set(model.parameters()["input"]) == {"weight"}
    assert set(model.parameters()["product"]) == {"weight"}
    assert set(model.parameters()["output"]) == {"weight", "bias"}
    assert "_output_mask" not in model.parameters()["product"]


def test_nested_model_freeze_unfreeze_and_training_mode_propagate() -> None:
    model = _make_model(compile_layers=False)
    model.input.freeze()
    trainable = model.trainable_parameters()
    assert trainable["input"] == {}
    assert set(trainable["product"]) == {"weight"}
    assert set(trainable["output"]) == {"weight", "bias"}
    model.input.unfreeze()
    assert set(model.trainable_parameters()["input"]) == {"weight"}
    model.eval()
    assert not model.training and not model.input.training and not model.product.training and not model.output.training
    model.train()
    assert model.training and model.input.training and model.product.training and model.output.training


def test_optimizer_step_updates_nested_parameters_and_reduces_loss() -> None:
    mx, nn = require_mlx()
    import mlx.optimizers as optim

    mx.random.seed(0)
    model = _make_model(compile_layers=True)
    vectors, edges, targets = _training_data()

    def loss():
        error = model(vectors, edges).array - targets
        return mx.mean(error * error)

    loss_and_grad = nn.value_and_grad(model, loss)
    optimizer = optim.Adam(learning_rate=0.03)
    initial = float(loss())
    initial_weight = model.product.weight
    for _ in range(30):
        value, gradients = loss_and_grad()
        optimizer.update(model, gradients)
        mx.eval(value, model.parameters(), optimizer.state)
    final = float(loss())
    assert final < initial * 0.2
    assert abs(float(mx.max(mx.abs(model.product.weight - initial_weight)))) > 1e-5


def test_trained_compiled_model_remains_rotation_invariant() -> None:
    mx, nn = require_mlx()
    import mlx.optimizers as optim

    model = _make_model(compile_layers=True)
    vectors, edges, targets = _training_data()

    def loss():
        error = model(vectors, edges).array - targets
        return mx.mean(error * error)

    loss_and_grad = nn.value_and_grad(model, loss)
    optimizer = optim.SGD(learning_rate=0.02)
    for _ in range(5):
        value, gradients = loss_and_grad()
        optimizer.update(model, gradients)
        mx.eval(value, model.parameters(), optimizer.state)

    alpha, beta, gamma = (mlx_backend.asarray(value) for value in (0.3, -0.4, 0.7))
    expected = model(vectors, edges).array
    actual = model(_rotate(vectors, alpha, beta, gamma), _rotate(edges, alpha, beta, gamma)).array
    assert abs(float(mx.max(mx.abs(actual - expected)))) < 2e-4


def test_save_and_load_weights_round_trip(tmp_path) -> None:
    mx, _ = require_mlx()
    model = _make_model(compile_layers=True)
    vectors, edges, _ = _training_data()
    expected = model(vectors, edges).array
    path = tmp_path / "model.npz"
    model.save_weights(str(path))

    restored = _make_model(compile_layers=True)
    restored.load_weights(str(path))
    actual = restored(vectors, edges).array
    mx.eval(expected, actual)
    assert abs(float(mx.max(mx.abs(actual - expected)))) < 1e-7


def test_parameterless_modules_nest_without_polluting_parameter_tree() -> None:
    _, nn = require_mlx()

    class Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = Gate("0e", "0e", "1o")
            self.norm = Norm()

    container = Container()
    assert set(container.children()) == {"gate", "norm"}
    assert container.parameters() == {"gate": {}, "norm": {}}
