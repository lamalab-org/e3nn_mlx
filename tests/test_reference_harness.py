from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from e3nn_core.cg import wigner_3j
from e3nn_core.wigner import so3_generators


ROOT = Path(__file__).resolve().parents[1]


def test_reference_generator_is_syntax_valid() -> None:
    source = ROOT / "tests" / "reference_generation" / "generate_e3nn_reference.py"
    ast.parse(source.read_text(), filename=str(source))


def test_reference_versions_are_pinned() -> None:
    requirements = (ROOT / "tests" / "reference_generation" / "requirements.txt").read_text().splitlines()
    assert requirements
    assert all("==" in line for line in requirements if line.strip() and not line.startswith("#"))


def test_checked_in_wigner_3j_matches_upstream_e3nn() -> None:
    reference_dir = ROOT / "tests" / "reference_data"
    manifest = json.loads((reference_dir / "manifest.json").read_text())
    assert manifest["schema"] == 1
    assert manifest["e3nn"] == "0.5.8"
    with np.load(reference_dir / "e3nn_p0_reference.npz") as fixture:
        for l1, l2, l3 in manifest["wigner_3j_triples"]:
            actual = np.asarray(wigner_3j(l1, l2, l3))
            expected = fixture[f"w3j_{l1}_{l2}_{l3}"]
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def _matrix_exp_generator(generator: np.ndarray, angles: np.ndarray) -> np.ndarray:
    scaled = np.remainder(angles, 2.0 * np.pi)[..., None, None] * generator / 64.0
    result = np.eye(generator.shape[-1]) + scaled
    term = scaled
    for order in range(2, 19):
        term = term @ scaled / order
        result = result + term
    for _ in range(6):
        result = result @ result
    return result


def test_generator_wigner_algorithm_matches_upstream_e3nn() -> None:
    reference_dir = ROOT / "tests" / "reference_data"
    with np.load(reference_dir / "e3nn_p0_reference.npz") as fixture:
        angles = fixture["angles"]
        for l in range(9):
            generators = np.asarray(so3_generators(l))
            actual = (
                _matrix_exp_generator(generators[1], angles[:, 0])
                @ _matrix_exp_generator(generators[0], angles[:, 1])
                @ _matrix_exp_generator(generators[1], angles[:, 2])
            )
            np.testing.assert_allclose(actual, fixture[f"wigner_d_{l}"], atol=2e-11, rtol=2e-11)


def _component_harmonics(vectors: np.ndarray, lmax: int) -> list[np.ndarray]:
    outputs = [np.ones((*vectors.shape[:-1], 1))]
    if lmax == 0:
        return outputs
    y1 = np.sqrt(3.0) * vectors
    outputs.append(y1)
    previous = y1
    for l in range(1, lmax):
        previous = (2 * l + 3) / np.sqrt(3.0 * (l + 1)) * np.einsum(
            "...a,...b,abc->...c", previous, y1, np.asarray(wigner_3j(l, 1, l + 1))
        )
        outputs.append(previous)
    return outputs


def test_cg_spherical_harmonics_recurrence_matches_upstream_e3nn() -> None:
    reference_dir = ROOT / "tests" / "reference_data"
    with np.load(reference_dir / "e3nn_p0_reference.npz") as fixture:
        vectors = fixture["vectors"]
        vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
        harmonics = _component_harmonics(vectors, 6)
        for l, component in enumerate(harmonics):
            np.testing.assert_allclose(component, fixture[f"sh_component_{l}"], atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(
                component / np.sqrt(2 * l + 1), fixture[f"sh_norm_{l}"], atol=2e-12, rtol=2e-12
            )
            np.testing.assert_allclose(
                component / np.sqrt(4 * np.pi), fixture[f"sh_integral_{l}"], atol=2e-12, rtol=2e-12
            )
