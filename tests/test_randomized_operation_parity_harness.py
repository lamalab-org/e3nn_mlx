"""Contracts for randomized parity across the public numerical surface."""

from __future__ import annotations

import numpy as np
import pytest

from evals.randomized_operation_parity import (
    KINDS,
    _mlx_worker,
    compare_results,
    generate_cases,
)


def test_randomized_operation_cases_are_reproducible_and_cover_every_family():
    first = generate_cases(seed=1729, cases_per_kind=2, max_l=3)
    second = generate_cases(seed=1729, cases_per_kind=2, max_l=3)
    assert first == second
    assert {case["kind"] for case in first} == set(KINDS)
    assert all(
        sum(case["kind"] == kind for case in first) == 2 for kind in KINDS
    )


def test_randomized_operation_values_are_finite_replayable_numpy_data():
    cases = generate_cases(seed=37, cases_per_kind=1, max_l=3)

    def arrays(value):
        if isinstance(value, dict):
            if set(value) == {"shape", "values"}:
                yield value
            else:
                for child in value.values():
                    yield from arrays(child)
        elif isinstance(value, list):
            for child in value:
                yield from arrays(child)

    for case in cases:
        for specification in arrays(case):
            value = np.asarray(specification["values"], dtype=np.float32)
            assert list(value.shape) == specification["shape"]
            assert np.isfinite(value).all()


def test_operation_comparison_retains_output_and_vjp_failures():
    case = generate_cases(
        seed=11,
        cases_per_kind=1,
        kinds=("norm",),
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
                "gradients": [{"shape": [1], "values": [2.0]}],
                "metadata": {},
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
                        "name": "mlx",
                        "status": "ok",
                        "shape": [1],
                        "output": [1.0],
                        "gradients": [{"shape": [1], "values": [3.0]}],
                        "metadata": {},
                    }
                ],
            }
        ],
    }
    report = compare_results([case], torch_result, mlx_result)
    assert report["failed"] == 1
    assert report["failures"][0]["case"] == case
    assert (
        report["rows"][0]["comparisons"][0]["gradient_max_abs"] == 1.0
    )


@pytest.mark.mlx
def test_generated_cases_execute_on_every_mlx_operation_variant():
    cases = generate_cases(seed=20260727, cases_per_kind=1, max_l=3)
    result = _mlx_worker(cases)
    assert result["backend"] == "mlx"
    for case, row in zip(cases, result["results"], strict=True):
        assert all(variant["status"] == "ok" for variant in row["variants"]), (
            case,
            row,
        )
