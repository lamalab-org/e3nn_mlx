"""Contract tests for randomized parity across core operation families."""

from __future__ import annotations

import numpy as np
import pytest

from evals.randomized_core_parity import (
    FAMILIES,
    _mlx_worker,
    compare_core_results,
    generate_core_cases,
)


def test_core_case_generation_is_seeded_and_covers_every_family():
    first = generate_core_cases(seed=8675309, cases_per_family=4, max_l=3)
    second = generate_core_cases(seed=8675309, cases_per_family=4, max_l=3)
    assert first == second
    assert len(first) == 4 * len(FAMILIES)
    assert {case["family"] for case in first} == set(FAMILIES)
    assert {
        case["kind"]
        for case in first
        if case["family"] == "tensor_product_wrappers"
    } == {"full", "fully_connected", "elementwise", "tensor_square"}


def test_core_cases_store_finite_replayable_numpy_values():
    cases = generate_core_cases(seed=101, cases_per_family=2, max_l=2)

    def arrays(value):
        if isinstance(value, dict):
            if set(value) == {"shape", "values"}:
                yield np.asarray(value["values"], dtype=np.float32).reshape(
                    value["shape"]
                )
            else:
                for child in value.values():
                    yield from arrays(child)
        elif isinstance(value, list):
            for child in value:
                yield from arrays(child)

    generated = [array for case in cases for array in arrays(case)]
    assert generated
    assert all(np.isfinite(array).all() for array in generated)


def test_core_comparison_retains_operation_and_replayable_failure():
    case = generate_core_cases(
        seed=7,
        cases_per_family=1,
        families=("norm",),
    )[0]
    torch_result = {
        "torch": "test",
        "e3nn": "test",
        "results": [
            {
                "id": case["id"],
                "status": "ok",
                "shape": [1],
                "output": [1.0],
            }
        ],
    }
    mlx_result = {
        "mlx": "test",
        "results": [
            {
                "id": case["id"],
                "variants": [
                    {
                        "name": "general",
                        "status": "ok",
                        "shape": [1],
                        "output": [2.0],
                    }
                ],
            }
        ],
    }
    report = compare_core_results([case], torch_result, mlx_result)
    assert report["failed"] == 1
    assert report["rows"][0]["family"] == "norm"
    assert report["failures"][0]["case"] == case


@pytest.mark.mlx
def test_one_generated_case_per_family_executes_on_all_mlx_variants():
    cases = generate_core_cases(seed=20260727, cases_per_family=1, max_l=3)
    result = _mlx_worker(cases)
    assert result["backend"] == "mlx"
    assert len(result["results"]) == len(FAMILIES)
    for row in result["results"]:
        assert all(variant["status"] == "ok" for variant in row["variants"]), row
