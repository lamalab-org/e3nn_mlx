"""Tests for the MLX spherical-harmonics animation data."""

from __future__ import annotations

import numpy as np
import pytest

from tutorials.spherical_harmonics_animation import (
    render_animation,
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


@pytest.mark.mlx
def test_transparent_animation_has_a_transparent_background(tmp_path) -> None:
    """The background must be transparent, and stay that way frame to frame."""

    pillow = pytest.importorskip("PIL.Image")
    output = render_animation(
        tmp_path / "sphharm.gif",
        lmax=1,
        resolution=8,
        frames=3,
        size_px=120,
        transparent=True,
    )
    image = pillow.open(output)

    # A GIF carries transparency as one reserved palette index, not an alpha
    # channel, so its presence is what makes the background see-through.
    # Read it before seeking: Pillow rewrites .info for each frame.
    transparent_index = image.info.get("transparency")
    assert transparent_index is not None
    assert image.n_frames == 3

    # The grabbed buffer is twice the figure size on a HiDPI backend; frames are
    # resampled so the asset does not depend on the rendering display.
    assert image.size == (120, 120)

    coverage = []
    for frame in range(image.n_frames):
        image.seek(frame)
        alpha = np.asarray(image.convert("RGBA"))[..., 3]
        assert all(alpha[y, x] == 0 for y in (0, -1) for x in (0, -1))
        coverage.append(float((alpha > 0).mean()))

    # Without disposal=2 each frame would paint over the last and opaque
    # coverage would climb toward the whole canvas.
    assert max(coverage) < 0.9

    # disposal=2 restores "the background colour", and a decoder that follows
    # the specification paints whatever index the logical screen descriptor
    # names. If that is not the transparent index the animation shows a solid
    # background in a browser even though Pillow reports it as transparent.
    header = output.read_bytes()[:13]
    assert header[11] == transparent_index


@pytest.mark.mlx
def test_opaque_animation_keeps_its_white_background(tmp_path) -> None:
    pillow = pytest.importorskip("PIL.Image")
    output = render_animation(
        tmp_path / "sphharm.gif", lmax=1, resolution=8, frames=2, size_px=120
    )
    image = pillow.open(output)
    assert image.info.get("transparency") is None
    corner = np.asarray(image.convert("RGB"))[0, 0]
    assert (corner > 240).all()
