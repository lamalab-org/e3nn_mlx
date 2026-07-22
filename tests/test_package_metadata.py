from __future__ import annotations

from importlib.metadata import version

import e3nn_mlx


def test_public_version_matches_distribution_metadata() -> None:
    assert e3nn_mlx.__version__ == version("e3nn-mlx")
    assert "__version__" in e3nn_mlx.__all__
