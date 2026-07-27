"""Executable contracts for claims made by the user-facing documentation."""

from __future__ import annotations

import ast
import importlib
from math import pi
from pathlib import Path
import re
import tomllib

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
    "upstream Linear default path order":
        "tests/test_upstream_o3_linear_norm.py::test_linear_default_weight_paths_follow_upstream_input_major_order",
    "grouped Linear external-weight gradients":
        "tests/test_upstream_o3_linear_norm.py::test_grouped_linear_external_weight_vjp_matches_blockwise_formula",
    "explicit tensor-product validation":
        "tests/test_tensor_product_compatibility_qualification.py::test_constructor_rejects_upstream_invalid_uvw_and_static_configurations",
    "tensor-product wrappers match explicit contractions":
        "tests/test_tensor_product_correctness.py::test_full_tensor_product_matches_explicit_tensor_product",
    "reduced-tensor intermediate and output filters":
        "tests/test_upstream_o3_reduce_tensor.py::test_upstream_reduced_tensor_supports_intermediate_and_output_filters",
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
    "generated and general tensor-product parity":
        "tests/test_tensor_product_correctness.py::test_optimized_full_tensor_product_matches_fallback_outputs_and_gradients",
    "generated and general scatter parity":
        "tests/test_metal_kernels.py::test_scatter_kernel_matches_general_forward_gradient_and_hessian",
    "general-path JVP support":
        "tests/test_documentation_contract.py::test_documented_general_paths_support_jvp",
    "equivariant graph-convolution recipe":
        "tests/test_documentation_contract.py::test_documented_graph_convolution_recipe_is_equivariant",
}


# Each entry anchors a concrete statement in a user-facing document to the test
# that enforces it.  The source excerpt is checked verbatim (after whitespace
# normalization), so deleting or materially rewriting a claim requires an
# explicit decision about its replacement evidence.
DOCUMENTED_SOURCE_CONTRACTS = (
    (
        "docs/COMPATIBILITY.md",
        "Golden reference data is generated with `e3nn==0.5.8`, `torch==2.7.1`, and `numpy==2.3.1`.",
        "tests/test_reference_harness.py::test_reference_versions_are_pinned",
    ),
    (
        "docs/COMPATIBILITY.md",
        "Integer angular momenta `0 <= l <= 6` are guaranteed by the compatibility baseline.",
        "tests/test_p0_equivariance.py::test_spherical_harmonics_equivariance_through_l6",
    ),
    (
        "docs/COMPATIBILITY.md",
        "Euler angles use upstream e3nn's active YXY convention `R = Ry(alpha) Rx(beta) Ry(gamma)`.",
        "tests/test_documentation_contract.py::test_p0_rotation_shape_and_active_yxy_convention",
    ),
    (
        "docs/COMPATIBILITY.md",
        "Leading input and unshared-weight dimensions follow NumPy broadcasting.",
        "tests/test_p0_equivariance.py::test_tensor_product_broadcasts_all_leading_dimensions",
    ),
    (
        "docs/COMPATIBILITY.md",
        "Mixed dtypes are rejected rather than applying implicit promotion.",
        "tests/test_documentation_contract.py::test_p0_tensor_product_dtype_contract",
    ),
    (
        "docs/COMPATIBILITY.md",
        "With `normalize=True`, spherical harmonics return the normalized scalar for `l=0` and zeros for `l>0` at the zero vector.",
        "tests/test_documentation_contract.py::test_p0_spherical_harmonic_normalization_and_zero_contract",
    ),
    (
        "docs/COMPATIBILITY.md",
        "`e3nn_core` remains importable without MLX.",
        "tests/test_nn_import_policy.py::test_nn_namespace_imports_without_site_packages",
    ),
    (
        "docs/guide/tensor_products.md",
        "`filter_ir_mid` restricts every sequential Clebsch--Gordan coupling, including the final coupling.",
        "tests/test_upstream_o3_reduce_tensor.py::test_upstream_reduced_tensor_supports_intermediate_and_output_filters",
    ),
    (
        "docs/HIGH_LEVEL_API.md",
        "High-level modules preserve the caller's style",
        "tests/test_high_level_api.py::test_o3_linear_raw_and_typed_calls_are_exact_and_compile",
    ),
    (
        "docs/HIGH_LEVEL_API.md",
        "Mixed raw and typed inputs to a binary tensor product return a typed result",
        "tests/test_documentation_contract.py::test_documented_zero_copy_unwrap_and_mixed_tensor_product_dispatch",
    ),
    (
        "docs/HIGH_LEVEL_API.md",
        "Their explicit `forward` methods and callable behavior are equivalent.",
        "tests/test_high_level_api.py::test_nn_modules_accept_raw_arrays_without_changing_typed_behavior",
    ),
    (
        "docs/HIGH_LEVEL_API.md",
        "Models imported from `e3nn_mlx.nn.models` return raw MLX arrays when their node features and attributes are raw.",
        "tests/test_high_level_api.py::test_v2106_network_compatibility_path_returns_raw_but_legacy_path_stays_typed",
    ),
    (
        "docs/HIGH_LEVEL_API.md",
        "No array copy is introduced.",
        "tests/test_documentation_contract.py::test_documented_zero_copy_unwrap_and_mixed_tensor_product_dispatch",
    ),
    (
        "docs/GATE_POINTS_2102.md",
        "eager, directed, loop-free, batched radius graphs",
        "tests/test_graph_radial.py::test_radius_graph_exact_batching_symmetry_and_rigid_motion_invariance",
    ),
    (
        "docs/GATE_POINTS_2102.md",
        "the exact upstream three-layer, 100-neuron model configuration",
        "tests/test_gate_points_2102_network.py::test_exact_upstream_network_constructs_and_runs_expected_shape",
    ),
    (
        "docs/GATE_POINTS_2102.md",
        "Position gradients still flow through edge vectors, lengths, radial bases, cutoffs, spherical harmonics, and every convolution.",
        "tests/test_gate_points_2102_network.py::test_network_position_feature_and_all_parameter_gradients_and_training_step",
    ),
    (
        "docs/V2106_POINT_MODELS.md",
        "Public model calls return `IrrepsArray` objects so the output representation remains explicit.",
        "tests/test_high_level_api.py::test_v2106_network_compatibility_path_returns_raw_but_legacy_path_stays_typed",
    ),
    (
        "docs/V2106_POINT_MODELS.md",
        "With `pool_nodes=True`, the result has one row per graph. With `pool_nodes=False`, it has one row per node.",
        "tests/test_v2106_networks.py::test_v2106_simple_network_batched_graphs_and_pooling_match_independent_runs",
    ),
    (
        "docs/V2106_POINT_MODELS.md",
        "Both network classes expose `forward_with_edges` to keep the entire differentiable calculation inside a compiled MLX graph once a topology is known.",
        "tests/test_v2106_networks.py::test_v2106_network_fixed_topology_compiles_and_reuses",
    ),
    (
        "docs/V2106_POINT_MODELS.md",
        "`alpha` starts at zero",
        "tests/test_v2106_convolution.py::test_v2106_convolution_construction_scalar_path_and_zero_alpha_initialization",
    ),
    (
        "docs/V2106_POINT_MODELS.md",
        "Empty edge lists are supported by every layer.",
        "tests/test_v2106_message_passing.py::test_v2106_message_passing_empty_edges_and_deepcopy",
    ),
    (
        "docs/guide/installation.md",
        "`e3nn-mlx` requires Python 3.11 or newer.",
        "tests/test_documentation_contract.py::test_documentation_metadata_workflow_and_timing_contracts",
    ),
    (
        "docs/guide/installation.md",
        "Every public operation retains a general MLX path.",
        "tests/test_metal_fallbacks.py::test_requested_metal_operations_fall_back_without_metal",
    ),
    (
        "docs/guide/irreps.md",
        "The integer $l$ determines the component count $2l+1$.",
        "tests/test_documentation_contract.py::test_documented_irrep_metadata_contract",
    ),
    (
        "docs/guide/tensor_products.md",
        "emits every allowed output path and has no learned weights",
        "tests/test_tensor_product_wrappers.py::test_full_tensor_product_builds_unweighted_full_output",
    ),
    (
        "docs/guide/tensor_products.md",
        "With `shared_weights=False`, external weights have shape `(..., weight_numel)`",
        "tests/test_upstream_o3_tensor_product.py::test_upstream_tensor_product_unshared_weight_broadcast_and_validation",
    ),
    (
        "docs/guide/tensor_products.md",
        "Both paths implement the same contraction and normalization conventions.",
        "tests/test_tensor_product_correctness.py::test_optimized_full_tensor_product_matches_fallback_outputs_and_gradients",
    ),
    (
        "docs/guide/linear.md",
        "Without explicit instructions, `Linear` enumerates compatible paths in input-major, then output-index order, matching upstream e3nn.",
        "tests/test_upstream_o3_linear_norm.py::test_linear_default_weight_paths_follow_upstream_input_major_order",
    ),
    (
        "docs/guide/linear.md",
        "The grouped execution path preserves derivatives with respect to external weights, including repeated input and output irrep blocks.",
        "tests/test_upstream_o3_linear_norm.py::test_grouped_linear_external_weight_vjp_matches_blockwise_formula",
    ),
    (
        "docs/COMPATIBILITY.md",
        "Mode `uvw` always requires weights",
        "tests/test_tensor_product_compatibility_qualification.py::test_constructor_rejects_upstream_invalid_uvw_and_static_configurations",
    ),
    (
        "docs/guide/equivariance.md",
        "actual = layer(x @ D_in.T)",
        "tests/test_documentation_contract.py::test_documented_linear_and_gate_examples_are_equivariant",
    ),
    (
        "docs/guide/performance.md",
        "The repository's `evals/` harness does this consistently across MLX, PyTorch CPU, and PyTorch MPS.",
        "tests/test_documentation_contract.py::test_documentation_metadata_workflow_and_timing_contracts",
    ),
    (
        "docs/guide/performance.md",
        "Disabling a kernel changes execution, not the mathematical operation.",
        "tests/test_metal_kernels.py::test_spherical_harmonics_kernel_matches_general_forward_gradient_and_hessian",
    ),
    (
        "docs/guide/performance.md",
        "Reverse-mode gradients and reverse-over-reverse second derivatives are supported by generated kernels.",
        "tests/test_metal_kernels.py::test_spherical_harmonics_kernel_matches_general_forward_gradient_and_hessian",
    ),
    (
        "docs/guide/performance.md",
        "should use the general path",
        "tests/test_documentation_contract.py::test_documented_general_paths_support_jvp",
    ),
    (
        "README.md",
        "the scatter fallback requires eager, fixed indices",
        "tests/test_documentation_contract.py::test_documented_general_paths_support_jvp",
    ),
    (
        "docs/guide/performance.md",
        "`scatter_sum(..., use_custom_kernel=False, jvp_safe=True)`",
        "tests/test_documentation_contract.py::test_documented_general_paths_support_jvp",
    ),
    (
        "docs/examples/convolution.md",
        "Distance-derived weights are rotation invariant; spherical harmonics and the tensor product carry the directional transformation law.",
        "tests/test_documentation_contract.py::test_documented_graph_convolution_recipe_is_equivariant",
    ),
    (
        "docs/examples/point_models.md",
        "The radial embedding, spherical harmonics, message passing, gates, and reduction can then remain in the compiled MLX graph.",
        "tests/test_v2106_networks.py::test_v2106_network_fixed_topology_compiles_and_reuses",
    ),
    (
        "docs/examples/point_models.md",
        "The generic `radius_graph` helper computes Euclidean neighbors for the given coordinates and does not infer periodic images.",
        "tests/test_documentation_contract.py::test_documented_radius_graph_has_no_implicit_periodic_images",
    ),
)


def _maximum_error(first, second) -> float:
    return float(abs(first - second).max())


def _python_blocks(path: Path):
    text = path.read_text(encoding="utf-8")
    return re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL)


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _test_node_exists(node_id: str) -> bool:
    relative_path, function_name = node_id.split("::", 1)
    path = ROOT / relative_path
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
        for node in ast.walk(tree)
    )


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
    assert _test_node_exists(node_id), f"{claim!r} references missing test {node_id}"


@pytest.mark.parametrize(
    "relative_path,excerpt,node_id",
    DOCUMENTED_SOURCE_CONTRACTS,
    ids=[Path(item[0]).name + ":" + item[2].rsplit("::", 1)[-1] for item in DOCUMENTED_SOURCE_CONTRACTS],
)
def test_each_documented_source_claim_is_present_and_has_live_evidence(
    relative_path: str, excerpt: str, node_id: str
) -> None:
    path = ROOT / relative_path
    assert path.is_file(), f"documented contract source does not exist: {relative_path}"
    assert _normalized_text(excerpt) in _normalized_text(path.read_text(encoding="utf-8")), (
        f"documented contract excerpt changed without updating its evidence: {excerpt!r}"
    )
    assert _test_node_exists(node_id), f"documented contract references missing test {node_id}"


def test_documentation_internal_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    failures = []
    for path in DOCUMENTED_MARKDOWN:
        for target in link_pattern.findall(path.read_text(encoding="utf-8")):
            target = target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                failures.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not failures, "broken documentation links:\n" + "\n".join(failures)


def test_documentation_toctree_targets_resolve() -> None:
    failures = []
    for path in ROOT.joinpath("docs").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for body in re.findall(r"```\{toctree\}\s*\n(.*?)```", text, re.DOTALL):
            for line in body.splitlines():
                target = line.strip()
                if not target or target.startswith(":") or "://" in target:
                    continue
                base = path.parent / target
                candidates = (
                    base,
                    base.with_suffix(".md"),
                    base.with_suffix(".rst"),
                    base / "index.md",
                    base / "index.rst",
                )
                if not any(candidate.exists() for candidate in candidates):
                    failures.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not failures, "broken toctree targets:\n" + "\n".join(failures)


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


def test_documented_irrep_metadata_contract() -> None:
    from e3nn_mlx import o3

    scalar = o3.Irrep("0e")
    polar_vector = o3.Irrep("1o")
    axial_vector = o3.Irrep("1e")
    features = o3.Irreps("16x0e + 8x1o")

    assert scalar.dim == 1 and scalar.p == 1
    assert polar_vector.dim == 3 and polar_vector.p == -1
    assert axial_vector.dim == 3 and axial_vector.p == 1
    assert all(o3.Irrep((degree, 1)).dim == 2 * degree + 1 for degree in range(9))
    assert features.dim == 40


def test_documentation_metadata_workflow_and_timing_contracts() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    assert project["requires-python"] == ">=3.11"
    assert project["urls"]["Documentation"] == "https://lamalab-org.github.io/e3nn_mlx/"
    assert {item.split("[", 1)[0].split(">", 1)[0] for item in project["optional-dependencies"]["docs"]} >= {
        "furo",
        "myst-parser",
        "sphinx",
        "sphinx-copybutton",
    }
    dependencies = "\n".join(project["dependencies"])
    assert "sys_platform == 'darwin' and platform_machine == 'arm64'" in dependencies
    assert "sys_platform == 'linux'" in dependencies
    assert metadata["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]

    workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    assert "python -m sphinx -W --keep-going -b html docs docs/_build/html" in workflow
    assert "github.event_name != 'pull_request'" in workflow
    assert "actions/configure-pages@v5" in workflow
    assert "actions/upload-pages-artifact@v4" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "pages: write" in workflow and "id-token: write" in workflow
    assert "name: github-pages" in workflow

    tests_workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert 'E3NN_MLX_REQUIRE_RUNTIME: "1"' in tests_workflow

    mlx_timing = (ROOT / "evals" / "mlx_cases.py").read_text(encoding="utf-8")
    torch_timing = (ROOT / "evals" / "torch_cases.py").read_text(encoding="utf-8")
    runner = (ROOT / "evals" / "common.py").read_text(encoding="utf-8")
    assert "mx.eval(result)" in mlx_timing and "mx.synchronize()" in mlx_timing
    assert "torch.mps.synchronize()" in torch_timing
    assert "synchronize(result)" in runner


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
def test_documented_linear_and_gate_examples_are_equivariant() -> None:
    mx = mlx_backend._require()
    from e3nn_mlx import nn, o3

    layer = o3.Linear("3x0e + 2x1o", "4x0e + 3x1o")
    values = mx.random.normal((16, layer.irreps_in.dim))
    rotation = o3.rand_matrix()
    input_matrix = o3.irreps_wigner_d_from_matrix(layer.irreps_in, rotation)
    output_matrix = o3.irreps_wigner_d_from_matrix(layer.irreps_out, rotation)
    actual = layer(values @ input_matrix.T)
    expected = layer(values) @ output_matrix.T
    assert _maximum_error(actual, expected) < 2e-5

    gate = nn.Gate("8x0e", [mx.tanh], "2x0e", [mx.sigmoid], "2x1o")
    gate_values = mx.random.normal((16, gate.irreps_in.dim))
    gate_input_matrix = o3.irreps_wigner_d_from_matrix(gate.irreps_in, rotation)
    gate_output_matrix = o3.irreps_wigner_d_from_matrix(gate.irreps_out, rotation)
    actual_gate = gate(gate_values @ gate_input_matrix.T)
    expected_gate = gate(gate_values) @ gate_output_matrix.T
    assert _maximum_error(actual_gate, expected_gate) < 2e-5


@pytest.mark.mlx
def test_documented_general_paths_support_jvp() -> None:
    mx = mlx_backend._require()
    from e3nn_mlx import o3, scatter_sum

    vectors = mx.array([[0.2, -0.3, 0.7], [-0.4, 0.1, 0.6]], dtype=mx.float32)
    vector_direction = mx.ones_like(vectors) * 0.1
    (harmonics,), (harmonics_tangent,) = mx.jvp(
        lambda value: o3.spherical_harmonics(
            [0, 1, 2], value, use_custom_kernel=False
        ),
        (vectors,),
        (vector_direction,),
    )
    assert harmonics.shape == harmonics_tangent.shape == (2, 9)

    source = mx.arange(8, dtype=mx.float32).reshape(4, 2)
    index = mx.array([0, 1, 0, 1], dtype=mx.int32)
    (scattered,), (scatter_tangent,) = mx.jvp(
        lambda value: scatter_sum(
            value,
            index,
            2,
            use_custom_kernel=False,
            jvp_safe=True,
        ),
        (source,),
        (mx.ones_like(source),),
    )
    expected_scatter = scatter_sum(source, index, 2)
    expected_tangent = scatter_sum(mx.ones_like(source), index, 2)
    assert _maximum_error(scattered, expected_scatter) == 0.0
    assert _maximum_error(scatter_tangent, expected_tangent) == 0.0

    product = o3.FullTensorProduct("1o", "1o", use_custom_kernel=True)
    right = mx.array([[-0.5, 0.1, 0.7], [0.2, 0.4, -0.1]], dtype=mx.float32)
    (product_value,), (product_tangent,) = mx.jvp(
        lambda left: product.differentiable_arrays(left, right),
        (vectors,),
        (vector_direction,),
    )
    assert product_value.shape == product_tangent.shape
    assert product_value.shape == (2, product.irreps_out.dim)
    mx.eval(harmonics_tangent, scatter_tangent, product_tangent)
    assert bool(mx.all(mx.isfinite(harmonics_tangent)))
    assert bool(mx.all(mx.isfinite(scatter_tangent)))
    assert bool(mx.all(mx.isfinite(product_tangent)))


@pytest.mark.mlx
def test_documented_graph_convolution_recipe_is_equivariant() -> None:
    mx = mlx_backend._require()
    from e3nn_mlx import math, o3, scatter_sum

    irreps_node = o3.Irreps("2x0e + 1x1o")
    irreps_sh = o3.Irreps.spherical_harmonics(2)
    product = o3.FullyConnectedTensorProduct(
        irreps_node,
        irreps_sh,
        irreps_node,
        internal_weights=False,
        shared_weights=False,
        use_custom_kernel=False,
    )
    positions = mx.array(
        [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [-0.2, 0.8, 0.1], [0.1, -0.3, 0.9]],
        dtype=mx.float32,
    )
    node_features = mx.random.normal((4, irreps_node.dim))
    edge_src = mx.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=mx.int32)
    edge_dst = mx.array([1, 2, 0, 3, 0, 3, 1, 2], dtype=mx.int32)

    def convolution(pos, features):
        edge_vectors = pos[edge_dst] - pos[edge_src]
        edge_lengths = mx.linalg.norm(edge_vectors, axis=-1)
        edge_attributes = o3.spherical_harmonics(
            irreps_sh,
            edge_vectors,
            normalize=True,
            normalization="component",
            use_custom_kernel=False,
        )
        radial = math.soft_one_hot_linspace(
            edge_lengths,
            start=0.0,
            end=2.0,
            number=5,
            basis="smooth_finite",
            cutoff=True,
        )
        invariant = mx.mean(radial, axis=-1, keepdims=True)
        weights = mx.broadcast_to(
            invariant, (edge_vectors.shape[0], product.weight_numel)
        )
        messages = product(features[edge_src], edge_attributes, weights)
        return scatter_sum(
            messages, edge_dst, pos.shape[0], use_custom_kernel=False
        )

    rotation = o3.rand_matrix()
    representation = o3.irreps_wigner_d_from_matrix(irreps_node, rotation)
    baseline = convolution(positions, node_features)
    actual = convolution(positions @ rotation.T, node_features @ representation.T)
    expected = baseline @ representation.T
    mx.eval(actual, expected)
    assert _maximum_error(actual, expected) < 3e-4


@pytest.mark.mlx
def test_documented_radius_graph_has_no_implicit_periodic_images() -> None:
    mx = mlx_backend._require()
    periodic_boundary_pair = mx.array(
        [[0.05, 0.0, 0.0], [9.95, 0.0, 0.0]], dtype=mx.float32
    )
    assert e3nn_mlx.radius_graph(periodic_boundary_pair, 0.2).shape == (2, 0)

    euclidean_pair = mx.array(
        [[0.05, 0.0, 0.0], [0.15, 0.0, 0.0]], dtype=mx.float32
    )
    edges = e3nn_mlx.radius_graph(euclidean_pair, 0.2)
    assert edges.shape == (2, 2)


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
