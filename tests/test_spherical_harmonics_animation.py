"""Tests for the MLX spherical-harmonics animation data."""

from __future__ import annotations

import numpy as np
import pytest

from tutorials.spherical_harmonics_animation import (
    spherical_harmonic_centers,
    spherical_harmonic_surfaces,
)


@pytest.mark.mlx
def test_animation_contains_every_harmonic_as_a_finite_radial_surface() -> None:
    coordinates, values, centers = spherical_harmonic_surfaces(
        lmax=3, resolution=8
    )

    assert coordinates.shape == (16, 16, 8, 3)
    assert values.shape == (16, 16, 8)
    assert centers.shape == (16, 3)
    assert np.isfinite(coordinates).all()
    assert np.isfinite(values).all()

    radii = np.linalg.norm(coordinates, axis=-1)
    radius_scale = 0.5 * np.sqrt(4.0 * np.pi) / np.sqrt(7.0)
    assert np.allclose(radii, radius_scale * np.abs(values), atol=2e-6)


def test_animation_centers_use_reference_l_m_square_placement() -> None:
    centers = spherical_harmonic_centers(2)
    assert np.array_equal(
        centers[:4],
        np.asarray(
            [(-1, 0, 1), (-1, 0, 0), (0, 0, 0), (0, 0, 1)],
            dtype=np.float32,
        ),
    )
    assert set(map(tuple, centers[:, (0, 2)])) == {
        (-1.0, -1.0),
        (-1.0, 0.0),
        (-1.0, 1.0),
        (0.0, -1.0),
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, -1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    }


@pytest.mark.mlx
def test_animation_rotates_coefficients_about_reference_axis() -> None:
    base, _, _ = spherical_harmonic_surfaces(lmax=2, resolution=8)
    quarter_turn, _, _ = spherical_harmonic_surfaces(
        lmax=2, resolution=8, angle=np.pi / 2
    )
    full_turn, _, _ = spherical_harmonic_surfaces(
        lmax=2, resolution=8, angle=2 * np.pi
    )
    assert not np.allclose(base[1:], quarter_turn[1:], atol=1e-4)
    assert np.allclose(base, full_turn, atol=2e-5)


def test_animation_surface_arguments_are_validated() -> None:
    with pytest.raises(ValueError, match="lmax"):
        spherical_harmonic_surfaces(lmax=-1)
    with pytest.raises(ValueError, match="resolution"):
        spherical_harmonic_surfaces(resolution=2)
