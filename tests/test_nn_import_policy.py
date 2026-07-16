"""Import-policy regression tests for optional NN runtime dependencies."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_nn_namespace_imports_without_site_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import e3nn_mlx; assert e3nn_mlx.SO3Activation.__name__ == 'SO3Activation'",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
