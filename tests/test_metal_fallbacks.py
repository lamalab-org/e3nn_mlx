"""Metal requests must remain portable to MLX's CPU backend."""

from __future__ import annotations

import pytest

import e3nn_mlx as e3nn
from e3nn_mlx.backend import mlx_backend
import e3nn_mlx.graph as graph_module
import e3nn_mlx.ops_sh as sh_module
import e3nn_mlx.ops_tp as tp_module


def _maximum_error(first, second) -> float:
    return float(abs(first - second).max())


@pytest.mark.mlx
def test_requested_metal_operations_fall_back_without_metal(monkeypatch) -> None:
    mx = mlx_backend._require()
    monkeypatch.setattr(graph_module, "mlx_metal_available", lambda: False)
    monkeypatch.setattr(sh_module, "mlx_metal_available", lambda: False)
    monkeypatch.setattr(tp_module, "mlx_metal_available", lambda: False)

    vectors = mx.array([[0.2, -0.3, 0.7], [-0.4, 0.1, 0.6]], dtype=mx.float32)
    requested_sh = e3nn.spherical_harmonics([0, 1, 2], vectors)
    general_sh = e3nn.spherical_harmonics(
        [0, 1, 2], vectors, use_custom_kernel=False
    )
    assert _maximum_error(requested_sh, general_sh) == 0.0

    source = mx.arange(12, dtype=mx.float32).reshape(3, 4)
    index = mx.array([0, 1, 0], dtype=mx.int32)
    requested_scatter = e3nn.scatter_sum(
        source, index, 2, use_custom_kernel=True
    )
    general_scatter = e3nn.scatter_sum(
        source, index, 2, use_custom_kernel=False
    )
    assert _maximum_error(requested_scatter, general_scatter) == 0.0

    requested_tp = e3nn.FullTensorProduct("1o", "1o")
    general_tp = e3nn.FullTensorProduct("1o", "1o", use_custom_kernel=False)
    assert requested_tp._metal_operation is None
    left = mx.array([[0.2, 0.4, -0.3]], dtype=mx.float32)
    right = mx.array([[-0.5, 0.1, 0.7]], dtype=mx.float32)
    left = e3nn.IrrepsArray("1o", left)
    right = e3nn.IrrepsArray("1o", right)
    assert (
        _maximum_error(
            requested_tp(left, right).array,
            general_tp(left, right).array,
        )
        == 0.0
    )
