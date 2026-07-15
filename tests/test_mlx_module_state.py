from __future__ import annotations

import pytest

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.compat import require_mlx
from e3nn_mlx.irreps_array import IrrepsArray
from e3nn_mlx.nn_linear import Linear
from e3nn_mlx.nn_gate import Gate
from e3nn_mlx.nn_norm import Norm
from e3nn_mlx.ops_tp import (
    ElementwiseTensorProduct,
    FullTensorProduct,
    FullyConnectedTensorProduct,
    TensorProduct,
    TensorSquare,
)


pytestmark = pytest.mark.mlx


def test_linear_and_norm_are_real_mlx_modules() -> None:
    _, nn = require_mlx()
    linear = Linear("2x0e+1o", "0e+1o")
    norm = Norm()
    assert isinstance(linear, nn.Module)
    assert isinstance(norm, nn.Module)
    assert set(linear.parameters()) == {"weight", "bias"}
    assert norm.parameters() == {}


def test_linear_parameter_update_changes_forward_result() -> None:
    linear = Linear("2x0e", "0e", bias=True)
    array = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    linear.update({"weight": mlx_backend.asarray([1.0, 0.0]), "bias": mlx_backend.asarray([0.0])})
    first = linear(array).array
    linear.update({"weight": mlx_backend.asarray([0.0, 2.0]), "bias": mlx_backend.asarray([1.0])})
    second = linear(array).array
    assert first.tolist() == [[2.0]]
    assert second.tolist() == [[7.0]]


def test_linear_only_biases_invariant_even_scalars() -> None:
    even = Linear("0e", "0e", bias=True)
    odd = Linear("0o", "0o", bias=True)
    mixed = Linear("0e+0o", "0e+0o", bias=True)
    assert even.bias is not None and even.bias.shape == (1,)
    assert odd.bias is None
    assert mixed.bias is not None and mixed.bias.shape == (1,)
    assert set(odd.parameters()) == {"weight"}


def test_weightless_linear_has_no_empty_trainable_parameter() -> None:
    linear = Linear("1o", "0e", bias=False)
    assert linear.weight is None
    assert linear.parameters() == {}
    output = linear(IrrepsArray("1o", mlx_backend.asarray([[1.0, 2.0, 3.0]])))
    assert output.array.tolist() == [[0.0]]


def test_linear_bias_broadcasts_for_unbatched_and_multibatch_inputs() -> None:
    linear = Linear("0e", "0e", bias=True)
    linear.update({"weight": mlx_backend.asarray([2.0]), "bias": mlx_backend.asarray([0.5])})
    unbatched = linear(IrrepsArray("0e", mlx_backend.asarray([3.0])))
    multibatch = linear(IrrepsArray("0e", mlx_backend.asarray([[[1.0], [2.0]], [[3.0], [4.0]]])))
    assert unbatched.array.tolist() == [6.5]
    assert multibatch.array.tolist() == [[[2.5], [4.5]], [[6.5], [8.5]]]


def test_linear_freeze_and_unfreeze_control_trainable_parameters() -> None:
    linear = Linear("0e", "0e", bias=True)
    linear.freeze(keys="weight", strict=True)
    assert set(linear.parameters()) == {"weight", "bias"}
    assert set(linear.trainable_parameters()) == {"bias"}
    linear.unfreeze(keys="weight", strict=True)
    assert set(linear.trainable_parameters()) == {"weight", "bias"}


def test_nn_value_and_grad_follows_linear_parameter_tree() -> None:
    mx, nn = require_mlx()
    linear = Linear("2x0e", "0e", bias=True)
    array = IrrepsArray("2x0e", mlx_backend.asarray([[1.0, -2.0], [0.5, 3.0]]))

    def loss():
        output = linear(array).array
        return mx.mean(output * output)

    value, gradients = nn.value_and_grad(linear, loss)()
    mx.eval(value, gradients)
    assert set(gradients) == {"weight", "bias"}
    assert gradients["weight"].shape == linear.weight.shape
    assert gradients["bias"].shape == linear.bias.shape


def test_gate_is_parameterless_mlx_module() -> None:
    _, nn = require_mlx()
    gate = Gate("0e", "0e", "1o")
    assert isinstance(gate, nn.Module)
    assert gate.parameters() == {}


def test_odd_gate_changes_output_parity_and_is_inversion_equivariant() -> None:
    mx, _ = require_mlx()
    gate = Gate("", "0o", "1o")
    assert str(gate.irreps_in) == "0o+1o"
    assert str(gate.irreps_out) == "1e"
    array = IrrepsArray("0o+1o", mlx_backend.asarray([[0.7, 1.0, -2.0, 3.0]]))
    inverted_input = IrrepsArray(array.irreps, -array.array)
    output = gate(array)
    inverted_output = gate(inverted_input)
    assert abs(float(mx.max(mx.abs(inverted_output.array - output.array)))) < 1e-6


def test_odd_scalar_default_activation_is_inversion_equivariant() -> None:
    mx, _ = require_mlx()
    gate = Gate("0o", "", "")
    positive = gate(IrrepsArray("0o", mlx_backend.asarray([[0.8]])))
    negative = gate(IrrepsArray("0o", mlx_backend.asarray([[-0.8]])))
    assert abs(float(mx.max(mx.abs(negative.array + positive.array)))) < 1e-7


def test_gate_uses_parity_specific_activations() -> None:
    gate = Gate(
        "0e+0o",
        "0e+0o",
        "1o+1o",
        even_scalar_activation=lambda x: x + 1,
        odd_scalar_activation=lambda x: 2 * x,
        even_gate_activation=lambda x: 3 * x,
        odd_gate_activation=lambda x: 4 * x,
    )
    array = IrrepsArray(
        "0e+0o+0e+0o+1o+1o",
        mlx_backend.asarray([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0]]),
    )
    output = gate(array)
    assert str(output.irreps) == "0e+0o+1o+1e"
    assert output.array.tolist() == [[2.0, 4.0, 9.0, 18.0, 27.0, -16.0, -32.0, -48.0]]


def test_gate_rejects_invalid_blocks_and_activation_configuration() -> None:
    with pytest.raises(ValueError, match="must be scalar"):
        Gate("", "1o", "1o")
    with pytest.raises(ValueError, match="match the gated multiplicity"):
        Gate("", "2x0e", "1o")
    with pytest.raises(ValueError, match="cannot be combined"):
        Gate("0e", "", "", scalar_activation=lambda x: x, even_scalar_activation=lambda x: x)


def test_tensor_product_family_are_mlx_modules() -> None:
    _, nn = require_mlx()
    modules = [
        TensorProduct("1o", "1o", "0e", [(0, 0, 0, "uvw", True)], compile_left_right=False),
        FullyConnectedTensorProduct("1o", "1o", "0e", compile_left_right=False),
        FullTensorProduct("1o", "1o", compile_left_right=False),
        ElementwiseTensorProduct("1o", "1o", compile_left_right=False),
        TensorSquare("1o", compile_left_right=False),
    ]
    assert all(isinstance(module, nn.Module) for module in modules)


def test_tensor_product_parameter_trees_distinguish_weight_ownership() -> None:
    internal = FullyConnectedTensorProduct("2x0e", "3x0e", "2x0e", compile_left_right=False)
    external = FullyConnectedTensorProduct(
        "2x0e", "3x0e", "2x0e", internal_weights=False, compile_left_right=False
    )
    full = FullTensorProduct("2x0e", "3x0e", compile_left_right=False)
    assert set(internal.parameters()) == {"weight"}
    assert internal.weight.shape == (internal.weight_numel,)
    assert external.parameters() == {} and external.weight is None
    assert full.parameters() == {} and full.weight is None
    assert "output_mask" not in internal.parameters()
    assert "_output_mask" not in internal.parameters()


def test_tensor_product_parameter_update_changes_compiled_forward() -> None:
    product = FullyConnectedTensorProduct("0e", "0e", "0e", compile_left_right=True)
    left = IrrepsArray("0e", mlx_backend.asarray([[2.0]]))
    right = IrrepsArray("0e", mlx_backend.asarray([[3.0]]))
    product.update({"weight": mlx_backend.asarray([1.0])})
    first = product(left, right).array
    product.update({"weight": mlx_backend.asarray([-2.0])})
    second = product(left, right).array
    assert first.tolist() == [[6.0]]
    assert second.tolist() == [[-12.0]]


def test_tensor_product_value_and_grad_matches_parameter_tree() -> None:
    mx, nn = require_mlx()
    product = FullyConnectedTensorProduct("2x0e", "2x0e", "0e", compile_left_right=False)
    left = IrrepsArray("2x0e", mlx_backend.asarray([[1.0, -2.0], [0.5, 1.5]]))
    right = IrrepsArray("2x0e", mlx_backend.asarray([[0.7, 0.2], [-1.0, 2.0]]))

    def loss():
        output = product(left, right).array
        return mx.mean(output * output)

    value, gradients = nn.value_and_grad(product, loss)()
    mx.eval(value, gradients)
    assert set(gradients) == {"weight"}
    assert gradients["weight"].shape == product.weight.shape


def test_tensor_product_freezing_and_weight_views_follow_module_weight() -> None:
    product = FullyConnectedTensorProduct("2x0e", "2x0e", "0e", compile_left_right=False)
    replacement = mlx_backend.asarray([float(index) for index in range(product.weight_numel)])
    product.update({"weight": replacement})
    assert product.weight_view_for_instruction(0).reshape(-1).tolist() == replacement.tolist()
    product.freeze(keys="weight", strict=True)
    assert set(product.parameters()) == {"weight"}
    assert product.trainable_parameters() == {}
    product.unfreeze(keys="weight", strict=True)
    assert set(product.trainable_parameters()) == {"weight"}


def test_tensor_square_parameter_state_matches_mode() -> None:
    unweighted = TensorSquare("2x1o", compile_left_right=False)
    weighted = TensorSquare("2x1o", "2x0e+2x2e", compile_left_right=False)
    external = TensorSquare("2x1o", "2x0e", internal_weights=False, compile_left_right=False)
    assert unweighted.parameters() == {}
    assert set(weighted.parameters()) == {"weight"}
    assert external.parameters() == {}
