"""Executable contracts for claims made by the user-facing documentation."""

from __future__ import annotations

import ast
import importlib
from math import pi
from pathlib import Path
import re

import pytest

import e3nn_mlx
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTED_MARKDOWN = tuple(
    sorted(
        {
            ROOT / "README.md",
            ROOT / "PORTING_MAP.md",
            ROOT / "CHANGELOG.md",
            ROOT / "benchmarks" / "README.md",
            ROOT / "tests" / "reference_generation" / "README.md",
            *ROOT.joinpath("docs").rglob("*.md"),
            *ROOT.joinpath("evals").glob("*.md"),
        }
    )
)

# Expensive claims are verified by their focused tests. This mapping makes the
# documentation-to-test relationship machine checked instead of leaving it as
# prose that can silently become stale.
DOCUMENTED_EVIDENCE = {
    "pinned upstream reference versions":
        "tests/test_reference_harness.py::test_reference_versions_are_pinned",
    "e3nn-compatible Wigner-3j coefficients":
        "tests/test_reference_harness.py::test_checked_in_wigner_3j_matches_upstream_e3nn",
    "canonical spherical-harmonic recurrence":
        "tests/test_reference_harness.py::test_cg_spherical_harmonics_recurrence_matches_upstream_e3nn",
    "O(3) metadata construction and arithmetic":
        "tests/test_upstream_o3_irreps_wigner.py::test_upstream_irreps_creation_arithmetic_and_properties",
    "Wigner matrices use Cartesian l=1 convention":
        "tests/test_upstream_o3_irreps_wigner.py::test_upstream_wigner_l1_is_cartesian_rotation",
    "P0 spherical-harmonic equivariance through l=6":
        "tests/test_p0_equivariance.py::test_spherical_harmonics_equivariance_through_l6",
    "tensor-product leading-dimension broadcasting":
        "tests/test_p0_equivariance.py::test_tensor_product_broadcasts_all_leading_dimensions",
    "generated-kernel CPU fallback":
        "tests/test_metal_fallbacks.py::test_requested_metal_operations_fall_back_without_metal",
    "raw and typed Linear dispatch":
        "tests/test_high_level_api.py::test_o3_linear_raw_and_typed_calls_are_exact_and_compile",
    "raw and typed tensor-product dispatch":
        "tests/test_high_level_api.py::test_o3_tensor_products_preserve_input_style",
    "high-level neural-network raw dispatch":
        "tests/test_high_level_api.py::test_nn_modules_accept_raw_arrays_without_changing_typed_behavior",
    "high-level compatibility layer delegates exactly":
        "tests/test_high_level_api.py::test_v2106_compatibility_convolution_matches_typed_core_exactly",
    "legacy model paths preserve typed results":
        "tests/test_high_level_api.py::test_v2106_network_compatibility_path_returns_raw_but_legacy_path_stays_typed",
    "high-level model equivariance":
        "tests/test_high_level_api.py::test_high_level_v2106_network_rotates_outputs_with_inputs",
    "tensor-product normalization modes":
        "tests/test_upstream_o3_tensor_product.py::test_upstream_fully_connected_tensor_product_is_statistically_normalized",
    "shared and unshared tensor-product weights":
        "tests/test_upstream_o3_tensor_product.py::test_upstream_tensor_product_unshared_weight_broadcast_and_validation",
    "tensor-product wrappers match explicit contractions":
        "tests/test_tensor_product_correctness.py::test_full_tensor_product_matches_explicit_tensor_product",
    "lazy package import without MLX":
        "tests/test_nn_import_policy.py::test_nn_namespace_imports_without_site_packages",
    "gate-points exact documented configuration":
        "tests/test_gate_points_2102_network.py::test_exact_upstream_network_constructs_and_runs_expected_shape",
    "gate-points E(3) equivariance":
        "tests/test_gate_points_2102_network.py::test_exact_upstream_network_e3_equivariance_rotation_inversion_translation",
    "gate-points fixed-topology compilation":
        "tests/test_gate_points_2102_network.py::test_network_fixed_topology_compiles_and_reuses_compiled_graph",
    "gate-points gradients and parameter update":
        "tests/test_gate_points_2102_network.py::test_network_position_feature_and_all_parameter_gradients_and_training_step",
    "gate-points convolution O(3) equivariance":
        "tests/test_gate_points_2102_convolution.py::test_convolution_o3_equivariance_for_rotations_and_inversion",
    "gate-points convolution gradients":
        "tests/test_gate_points_2102_convolution.py::test_convolution_input_and_parameter_gradients_are_finite_and_complete",
    "gate-points edge ordering and empty neighborhoods":
        "tests/test_gate_points_2102_convolution.py::test_convolution_edge_order_invariance_and_empty_neighborhood",
    "gate-points batched reduction":
        "tests/test_gate_points_2102_network.py::test_network_batched_graphs_equal_independent_evaluation_and_node_reduction",
    "gate-points optional inputs and isolated nodes":
        "tests/test_gate_points_2102_network.py::test_network_optional_inputs_isolated_nodes_and_nonreduced_output",
    "gate-points deepcopy and weight round trip":
        "tests/test_gate_points_2102_network.py::test_network_deepcopy_and_weight_round_trip",
    "radius graph directed, loop-free, batched behavior":
        "tests/test_graph_radial.py::test_radius_graph_exact_batching_symmetry_and_rigid_motion_invariance",
    "radial basis families compile and differentiate":
        "tests/test_graph_radial.py::test_soft_one_hot_all_bases_compile_shape_dtype_and_gradients",
    "scatter shape, compilation, and gradients":
        "tests/test_graph_radial.py::test_scatter_sum_values_multidimensional_empty_compile_and_gradient",
    "v2106 upstream-sized configurations":
        "tests/test_v2106_networks.py::test_v2106_exact_upstream_network_configurations_construct_and_run",
    "v2106 learned residual starts at zero":
        "tests/test_v2106_convolution.py::test_v2106_convolution_construction_scalar_path_and_zero_alpha_initialization",
    "v2106 convolution active residual equivariance":
        "tests/test_v2106_convolution.py::test_v2106_convolution_o3_equivariance_with_active_alpha_branch",
    "v2106 convolution edge ordering and empty edges":
        "tests/test_v2106_convolution.py::test_v2106_convolution_edge_order_and_empty_edges",
    "v2106 message-passing path filtering":
        "tests/test_v2106_message_passing.py::test_v2106_message_passing_upstream_configuration_and_path_filtering",
    "v2106 message-passing gradients":
        "tests/test_v2106_message_passing.py::test_v2106_message_passing_input_and_parameter_gradients",
    "v2106 message-passing empty edges":
        "tests/test_v2106_message_passing.py::test_v2106_message_passing_empty_edges_and_deepcopy",
    "v2106 E(3) equivariance":
        "tests/test_v2106_networks.py::test_v2106_network_e3_equivariance_rotation_inversion_translation",
    "v2106 pooling and batched graphs":
        "tests/test_v2106_networks.py::test_v2106_simple_network_batched_graphs_and_pooling_match_independent_runs",
    "v2106 fixed-topology compilation":
        "tests/test_v2106_networks.py::test_v2106_network_fixed_topology_compiles_and_reuses",
    "v2106 gradients and training":
        "tests/test_v2106_networks.py::test_v2106_network_all_input_and_parameter_gradients_and_training",
    "v2106 empty edges and aliases":
        "tests/test_v2106_networks.py::test_v2106_network_automatic_edges_alias_empty_edges_and_nonpooled_output",
    "v2106 deepcopy and weight round trip":
        "tests/test_v2106_networks.py::test_v2106_network_deepcopy_and_weight_round_trip",
    "reverse-mode and second derivatives for Metal kernels":
        "tests/test_metal_kernels.py::test_spherical_harmonics_kernel_matches_general_forward_gradient_and_hessian",
}


def _maximum_error(first, second) -> float:
    return float(abs(first - second).max())


def _python_blocks(path: Path):
    text = path.read_text(encoding="utf-8")
    return re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL)


def _autosummary_targets():
    targets = []
    for path in sorted(ROOT.joinpath("docs", "api").glob("*.rst")):
        module = None
        in_summary = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(".. currentmodule::"):
                module = stripped.split("::", 1)[1].strip()
                in_summary = False
            elif stripped == ".. autosummary::":
                in_summary = True
            elif in_summary and stripped and not stripped.startswith(":"):
                if line.startswith("   "):
                    targets.append((path, module, stripped))
                else:
                    in_summary = False
    return tuple(targets)


AUTOSUMMARY_TARGETS = _autosummary_targets()


@pytest.mark.parametrize("path", DOCUMENTED_MARKDOWN, ids=lambda path: path.name)
def test_documented_python_blocks_are_valid_python(path: Path) -> None:
    for index, block in enumerate(_python_blocks(path), start=1):
        compile(block, f"{path}:python-block-{index}", "exec")


@pytest.mark.parametrize("path,module_name,symbol", AUTOSUMMARY_TARGETS)
def test_every_generated_api_symbol_is_importable(
    path: Path, module_name: str | None, symbol: str
) -> None:
    assert module_name is not None, f"{path} lists {symbol} without currentmodule"
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), f"{path} documents missing {module_name}.{symbol}"


@pytest.mark.parametrize(
    "claim,node_id", DOCUMENTED_EVIDENCE.items(), ids=DOCUMENTED_EVIDENCE.keys()
)
def test_every_documented_behavior_claim_has_live_test_evidence(
    claim: str, node_id: str
) -> None:
    relative_path, function_name = node_id.split("::", 1)
    path = ROOT / relative_path
    assert path.is_file(), f"{claim!r} references missing test file {relative_path}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_name in functions, (
        f"{claim!r} references missing test function {node_id}"
    )


def test_documented_public_namespaces_and_exports_exist() -> None:
    from e3nn_mlx import math, nn, o3

    expected_o3 = {
        "Irrep", "Irreps", "IrrepsArray", "Linear", "Norm",
        "TensorProduct", "FullTensorProduct", "FullyConnectedTensorProduct",
        "ElementwiseTensorProduct", "TensorSquare", "ReducedTensorProducts",
        "SphericalHarmonics", "ToS2Grid", "FromS2Grid", "SO3Grid",
        "wigner_D",
    }
    expected_nn = {
        "Activation", "BatchNorm", "Dropout", "Gate", "Identity",
        "NormActivation", "Extract", "ExtractIr", "FullyConnectedNet",
        "S2Activation", "SO3Activation", "models",
    }
    expected_math = {"soft_one_hot_linspace", "soft_unit_step", "smooth_cutoff"}
    assert all(hasattr(o3, name) for name in expected_o3)
    assert all(hasattr(nn, name) for name in expected_nn)
    assert all(hasattr(math, name) for name in expected_math)
    assert o3.wigner_D is o3.wigner_d


def test_documented_model_import_paths_and_flat_aliases_exist() -> None:
    from e3nn_mlx.models.v2106 import (
        Convolution as CoreConvolution,
        MessagePassing as CoreMessagePassing,
        NetworkForAGraphWithAttributes as CoreAttributedNetwork,
        SimpleNetwork as CoreSimpleNetwork,
    )
    from e3nn_mlx.nn.models.v2106 import SimpleNetwork
    from e3nn_mlx.nn.models.v2106.gate_points_message_passing import MessagePassing
    from e3nn_mlx.nn.models.v2106.points_convolution import Convolution

    assert e3nn_mlx.V2106Convolution is CoreConvolution
    assert e3nn_mlx.V2106MessagePassing is CoreMessagePassing
    assert e3nn_mlx.SimpleNetwork is CoreSimpleNetwork
    assert e3nn_mlx.NetworkForAGraphWithAttributes is CoreAttributedNetwork
    assert callable(SimpleNetwork.forward_with_edges)
    assert callable(Convolution.forward_arrays)
    assert callable(MessagePassing.forward_arrays)


@pytest.mark.mlx
def test_readme_linear_example_and_raw_typed_contract_execute() -> None:
    mx = mlx_backend._require()
    from e3nn_mlx import o3

    linear = o3.Linear("16x0e + 16x1o", "32x0e + 16x1o")
    values = mx.random.normal((8, linear.irreps_in.dim))
    raw = linear(values)
    typed_input = o3.IrrepsArray(linear.irreps_in, values)
    typed = linear(typed_input)
    assert isinstance(raw, mx.array)
    assert isinstance(typed, o3.IrrepsArray)
    assert typed.irreps == linear.irreps_out
    assert typed_input.array is values
    assert _maximum_error(raw, typed.array) == 0.0


@pytest.mark.mlx
def test_documented_zero_copy_unwrap_and_mixed_tensor_product_dispatch() -> None:
    mx = mlx_backend._require()
    from e3nn_mlx import o3
    from e3nn_mlx._high_level import unwrap_irreps_arrays

    left = mx.random.normal((4, 3))
    right = mx.random.normal((4, 3))
    typed_left = IrrepsArray("1o", left)
    assert unwrap_irreps_arrays(typed_left, True) is left

    product = o3.FullTensorProduct("1o", "1o")
    mixed = product(typed_left, right)
    raw = product(left, right)
    assert isinstance(mixed, IrrepsArray)
    assert isinstance(raw, mx.array)
    assert _maximum_error(mixed.array, raw) == 0.0


@pytest.mark.mlx
def test_p0_rotation_shape_and_active_yxy_convention() -> None:
    mx = mlx_backend._require()
    alpha = mx.array([0.2, -0.4])
    beta = mx.array([0.3, 0.7])
    gamma = mx.array([-0.1, 0.5])
    rotation = e3nn_mlx.rotation_matrix(alpha, beta, gamma)
    expected = (
        e3nn_mlx.matrix_y(alpha)
        @ e3nn_mlx.matrix_x(beta)
        @ e3nn_mlx.matrix_y(gamma)
    )
    assert rotation.shape == (2, 3, 3)
    assert _maximum_error(rotation, expected) == 0.0
    determinant = (
        rotation[:, 0, 0]
        * (rotation[:, 1, 1] * rotation[:, 2, 2] - rotation[:, 1, 2] * rotation[:, 2, 1])
        - rotation[:, 0, 1]
        * (rotation[:, 1, 0] * rotation[:, 2, 2] - rotation[:, 1, 2] * rotation[:, 2, 0])
        + rotation[:, 0, 2]
        * (rotation[:, 1, 0] * rotation[:, 2, 1] - rotation[:, 1, 1] * rotation[:, 2, 0])
    )
    assert _maximum_error(determinant, mx.ones((2,))) < 2e-6

    vector = mx.array([[0.2, -0.3, 0.7], [-0.5, 0.1, 0.4]])
    transformed = vector[:, None, :] @ mx.swapaxes(rotation, -1, -2)
    assert transformed.shape == (2, 1, 3)


@pytest.mark.mlx
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_p0_spherical_harmonic_normalization_and_zero_contract(
    normalization: str,
) -> None:
    mx = mlx_backend._require()
    vectors = mx.array([[0.2, -0.3, 0.7]], dtype=mx.float32)
    vectors = vectors / mx.linalg.norm(vectors, axis=-1, keepdims=True)
    zero = mx.zeros((1, 3), dtype=mx.float32)
    harmonics = e3nn_mlx.spherical_harmonics(
        list(range(7)), vectors, normalization=normalization
    )
    at_zero = e3nn_mlx.spherical_harmonics(
        list(range(7)), zero, normalization=normalization
    )

    cursor = 0
    for degree in range(7):
        width = 2 * degree + 1
        block = harmonics[..., cursor : cursor + width]
        squared_norm = float(mx.sum(block * block))
        expected = {
            "component": float(width),
            "norm": 1.0,
            "integral": float(width) / (4.0 * pi),
        }[normalization]
        assert abs(squared_norm - expected) < 2e-4
        zero_block = at_zero[..., cursor : cursor + width]
        if degree == 0:
            assert _maximum_error(zero_block, block) < 2e-6
        else:
            assert _maximum_error(zero_block, mx.zeros_like(zero_block)) == 0.0
        cursor += width


@pytest.mark.mlx
def test_p0_tensor_product_dtype_contract() -> None:
    mx = mlx_backend._require()
    left = IrrepsArray("1o", mx.ones((2, 3), dtype=mx.float32))
    right = IrrepsArray("1o", mx.ones((2, 3), dtype=mx.float16))
    with pytest.raises(TypeError, match="matching dtypes"):
        e3nn_mlx.tensor_product(left, right)
