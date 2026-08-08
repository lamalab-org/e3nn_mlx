"""Contract tests for the seeded cross-framework qualification harness."""

from __future__ import annotations

import numpy as np
import pytest

from evals.randomized_tensor_product_parity import (
    MLX_VARIANTS,
    MODES,
    _mlx_worker,
    compare_results,
    generate_cases,
)


def test_randomized_tensor_product_cases_are_reproducible_and_structurally_broad():
    first = generate_cases(seed=1729, count=64, max_l=4, max_mul=4)
    second = generate_cases(seed=1729, count=64, max_l=4, max_mul=4)
    assert first == second
    assert {case["mode"] for case in first} == set(MODES)
    assert {case["weight_layout"] for case in first} >= {
        "scalar-batch",
        "shared-grid",
        "shared-vector",
        "unshared-batch",
        "unshared-singleton",
        "unshared-grid",
    }
    assert {case["irrep_normalization"] for case in first} == {
        "component",
        "norm",
    }
    assert {case["path_normalization"] for case in first} == {
        "element",
        "path",
    }
    assert any("0x" in case["irreps_in1"] for case in first)
    assert any(
        len({instruction[2] for instruction in case["instructions"]})
        < len(case["instructions"])
        for case in first
    )
    assert any(
        {instruction[4] for instruction in case["instructions"]} == {False, True}
        for case in first
    )


def test_randomized_case_values_are_numpy_replayable_float32_data():
    cases = generate_cases(seed=37, count=8)
    for case in cases:
        for name in ("left", "right", "weight"):
            specification = case[name]
            if specification is None:
                continue
            value = np.asarray(specification["values"], dtype=np.float32)
            assert list(value.shape) == specification["shape"]
            assert np.isfinite(value).all()


def test_comparison_reports_every_execution_variant_and_retains_failures():
    case = generate_cases(seed=11, count=1)[0]
    reference = np.asarray([[1.0, 2.0]], dtype=np.float32)
    torch_result = {
        "torch": "test",
        "results": [
            {
                "id": case["id"],
                "status": "ok",
                "shape": [1, 2],
                "output": reference.tolist(),
            }
        ],
    }
    variants = []
    for name, _, use_custom_kernel in MLX_VARIANTS:
        output = reference.copy()
        if use_custom_kernel:
            output[0, 0] += 0.1
        variants.append(
            {
                "name": name,
                "status": "ok",
                "shape": [1, 2],
                "output": output.tolist(),
                "custom_eligible": use_custom_kernel,
                "kernel_kind": (
                    "scalar_paths" if use_custom_kernel else None
                ),
            }
        )
    mlx_result = {
        "mlx": "test",
        "results": [{"id": case["id"], "variants": variants}],
    }
    report = compare_results(
        [case],
        torch_result,
        mlx_result,
        atol=1e-5,
        rtol=1e-5,
        norm_tolerance=1e-5,
    )
    assert report["passed"] == 0
    assert report["failed"] == 1
    assert [row["name"] for row in report["rows"][0]["comparisons"]] == [
        name for name, _, _ in MLX_VARIANTS
    ]
    assert report["failures"][0]["case"] == case


@pytest.mark.mlx
def test_generated_cases_execute_on_every_mlx_path():
    cases = generate_cases(seed=20260727, count=24, max_l=3, max_mul=3)
    result = _mlx_worker(cases)
    assert result["backend"] == "mlx"
    for row in result["results"]:
        assert len(row["variants"]) == len(MLX_VARIANTS)
        assert {
            variant["name"] for variant in row["variants"]
        } == {name for name, _, _ in MLX_VARIANTS}
        assert all(variant["status"] == "ok" for variant in row["variants"]), row
