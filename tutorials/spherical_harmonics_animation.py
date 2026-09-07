"""Rebuild e3nn's spherical-harmonics animation with MLX.

This is an MLX translation of e3nn's 2020 ``examples/animated_rsh.py``, which
produced the website asset. It retains the legacy basis, ``(l, m)`` placement,
coefficient-space rotation, viewing direction, radius scale, and color scale.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from math import pi, sqrt
from pathlib import Path

import mlx.core as mx
import numpy as np

import e3nn_mlx as o3


def spherical_harmonic_centers(lmax: int) -> np.ndarray:
    """Return the original animation's square-grid centers in e3nn order."""

    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    centers = np.asarray(
        [
            (
                degree + (order if order < 0 else 0) - lmax / 2.0,
                0,
                lmax / 2.0 - degree + (order if order > 0 else 0),
            )
            for degree in range(lmax + 1)
            for order in range(-degree, degree + 1)
        ],
        dtype=np.float32,
    )
    return centers


@lru_cache(maxsize=None)
def _surface_basis(lmax: int, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the website asset's z-polar grid and harmonic synthesis basis."""

    alphas = np.linspace(
        0.0, 2.0 * pi, 2 * resolution, dtype=np.float32
    )
    betas = np.linspace(0.0, pi, resolution, dtype=np.float32)
    alpha_grid, beta_grid = np.meshgrid(alphas, betas, indexing="ij")
    directions = np.stack(
        (
            np.sin(beta_grid) * np.cos(alpha_grid),
            np.sin(beta_grid) * np.sin(alpha_grid),
            np.cos(beta_grid),
        ),
        axis=-1,
    ).astype(np.float32)
    harmonics = o3.spherical_harmonics(
        list(range(lmax + 1)),
        mx.array(directions.reshape(-1, 3)),
        normalize=True,
        normalization="integral",
        use_custom_kernel=False,
    )
    basis = np.asarray(harmonics).reshape(
        2 * resolution, resolution, (lmax + 1) ** 2
    )
    return directions, basis


def spherical_harmonic_surfaces(
    *, lmax: int = 5, resolution: int = 50, angle: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local surfaces, signed values, and original ``(l, m)`` centers.

    The coefficient basis is rotated about ``(1, 0, 1)`` by ``angle``, exactly
    as in the reference generator. Surface/component order remains canonical
    e3nn order: increasing ``l``, then ``m = -l, ..., l``.
    """

    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    if resolution < 3:
        raise ValueError("resolution must be at least 3")

    directions, basis = _surface_basis(lmax, resolution)
    # The website asset predates e3nn's current Cartesian component
    # convention. Its l=1 basis is (-y, -z, -x). Express that improper basis
    # transformation in today's Torch-compatible basis without changing the
    # public e3nn-mlx spherical harmonics.
    legacy_basis = mx.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
        dtype=mx.float32,
    )
    irreps = o3.Irreps.spherical_harmonics(lmax)
    basis_change = o3.o3.irreps_wigner_d_from_matrix(
        irreps, legacy_basis
    )

    # Original code:
    # compose(0, -pi/4, 0, *compose(0, 0, angle, 0, pi/4, 0))
    zero = mx.array(0.0, dtype=mx.float32)
    tilt = mx.array(pi / 4.0, dtype=mx.float32)
    legacy_angles = o3.compose_angles(
        zero,
        -tilt,
        zero,
        *o3.compose_angles(
            zero, zero, mx.array(angle, dtype=mx.float32), zero, tilt, zero
        ),
    )
    legacy_rotation = o3.angles_to_matrix(*legacy_angles)
    current_rotation = (
        mx.swapaxes(legacy_basis, -1, -2)
        @ legacy_rotation
        @ legacy_basis
    )
    rotation = o3.o3.irreps_wigner_d_from_matrix(irreps, current_rotation)
    coefficients = basis_change @ rotation
    values = np.einsum(
        "abi,zi->zab", basis, np.asarray(coefficients), optimize=True
    ).astype(np.float32)

    radius_scale = np.float32(
        0.5 * sqrt(4.0 * pi) / sqrt(2 * lmax + 1)
    )
    coordinates = (
        radius_scale * np.abs(values)[..., None] * directions[None, ...]
    )
    return coordinates, values, spherical_harmonic_centers(lmax)


def _write_transparent_gif(
    path: Path, frames_rgba, duration_ms: int, size_px: int
) -> None:
    """Assemble RGBA frames into a GIF whose background is transparent.

    A GIF carries transparency as a single reserved palette index, so each
    frame is quantised to 255 colours and index 255 is kept for the mask.
    ``disposal=2`` clears each frame before the next is drawn; without it the
    transparent regions accumulate and earlier frames ghost through.

    Frames are resampled to ``size_px``. A HiDPI backend reports a device pixel
    ratio of 2, so the grabbed buffer is twice the requested figure size, and
    without this the output would silently depend on the display it was
    rendered on. Downsampling also supersamples the surface edges.
    """

    from PIL import Image

    images = []
    for array in frames_rgba:
        frame = Image.fromarray(array, "RGBA")
        if frame.size != (size_px, size_px):
            frame = frame.resize((size_px, size_px), Image.LANCZOS)
        alpha = frame.getchannel("A")
        indexed = frame.convert("RGB").quantize(colors=255)
        indexed.paste(255, alpha.point(lambda value: 255 if value <= 128 else 0))
        images.append(indexed)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        transparency=255,
        # disposal=2 means "restore to the background colour". Decoders differ:
        # Pillow restores to transparent, but a decoder that follows the
        # specification literally paints the colour the logical screen
        # descriptor names. That index defaults to 0, an arbitrary opaque
        # palette entry, which is what makes the animation appear on a solid
        # background in a browser. Point it at the transparent index so both
        # readings clear to nothing.
        background=255,
        disposal=2,
        optimize=False,
    )


def render_animation(
    output: str | Path,
    *,
    lmax: int = 5,
    resolution: int = 50,
    frames: int = 40,
    duration_ms: int = 30,
    size_px: int = 500,
    transparent: bool = False,
) -> Path:
    """Render the reference coefficient rotation as a GIF.

    ``transparent=True`` leaves the page showing through the figure instead of
    painting it white, so the animation sits on any documentation theme.
    """

    if frames < 1:
        raise ValueError("frames must be positive")
    if duration_ms < 1:
        raise ValueError("duration_ms must be positive")
    if size_px < 100:
        raise ValueError("size_px must be at least 100")

    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation, colors
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:  # pragma: no cover - depends on optional tools
        raise RuntimeError(
            "animation export requires matplotlib and Pillow"
        ) from exc

    dpi = 100
    figure = plt.figure(
        figsize=(size_px / dpi, size_px / dpi),
        dpi=dpi,
        facecolor="none" if transparent else "white",
    )
    if transparent:
        figure.patch.set_alpha(0.0)
    columns = lmax + 1
    centers = spherical_harmonic_centers(lmax)
    axes = []
    for center in centers:
        column = int(round(float(center[0] + lmax / 2.0)))
        row = int(round(float(lmax / 2.0 - center[2])))
        axis = figure.add_axes(
            (
                column / columns,
                1.0 - (row + 1) / columns,
                1.0 / columns,
                1.0 / columns,
            ),
            projection="3d",
        )
        # Independent square viewports preserve the same x/y/z scale for
        # every harmonic. Translating all surfaces into one Matplotlib 3D axis
        # visibly compressed one screen direction, unlike Plotly's renderer.
        axis.set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55), zlim=(-0.55, 0.55))
        axis.set_box_aspect((1, 1, 1), zoom=1.35)
        axis.set_proj_type("persp")
        # Reference camera: eye=(0, -1.3, 0), up=(0, 0, 1).
        axis.view_init(elev=0.0, azim=-90.0, roll=0.0)
        axis.set_axis_off()
        if transparent:
            # A 3D axis still paints its own background rectangle even with the
            # axis decorations switched off.
            axis.patch.set_alpha(0.0)
        axes.append(axis)
    color_map = LinearSegmentedColormap.from_list(
        "e3nn_bwr",
        ((0.0, (0 / 255, 50 / 255, 1.0)),
         (0.5, (200 / 255, 200 / 255, 200 / 255)),
         (1.0, (1.0, 50 / 255, 0 / 255))),
    )
    color_norm = colors.Normalize(vmin=-0.5, vmax=0.5, clip=True)
    artists = []

    def update(frame: int):
        while artists:
            artists.pop().remove()
        angle = 2.0 * pi * frame / frames
        coordinates, values, _ = spherical_harmonic_surfaces(
            lmax=lmax, resolution=resolution, angle=angle
        )
        for axis, surface, harmonic in zip(
            axes, coordinates, values, strict=True
        ):
            artists.append(
                axis.plot_surface(
                    surface[..., 0],
                    surface[..., 1],
                    surface[..., 2],
                    facecolors=color_map(color_norm(harmonic)),
                    linewidth=0,
                    antialiased=True,
                    shade=True,
                )
            )
        return tuple(artists)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if transparent:
        # PillowWriter flattens each frame onto an opaque canvas, so the frames
        # are grabbed directly and given a reserved transparent palette index.
        rendered = []
        for frame in range(frames):
            update(frame)
            figure.canvas.draw()
            rendered.append(
                np.asarray(figure.canvas.buffer_rgba()).copy()
            )
        _write_transparent_gif(output_path, rendered, duration_ms, size_px)
    else:
        movie = animation.FuncAnimation(
            figure, update, frames=frames, interval=duration_ms, blit=False
        )
        movie.save(
            output_path,
            # GIF delays are stored in centiseconds. An integer frame rate
            # avoids floating-point truncation turning 30 ms into 20 ms.
            writer=animation.PillowWriter(
                fps=max(1, round(1000.0 / duration_ms))
            ),
            dpi=dpi,
        )
    plt.close(figure)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild e3nn's spherical-harmonics GIF with MLX."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("sphharm_mlx.gif")
    )
    parser.add_argument("--lmax", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--duration-ms", type=int, default=30)
    parser.add_argument("--size-px", type=int, default=500)
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="leave the background transparent instead of white",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = render_animation(
        args.output,
        lmax=args.lmax,
        resolution=args.resolution,
        frames=args.frames,
        duration_ms=args.duration_ms,
        size_px=args.size_px,
        transparent=args.transparent,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
