from __future__ import annotations

import pytest

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray
from e3nn_mlx.nn_linear import Linear
from e3nn_mlx.ops_basic import ExtensionKernel, compile_or_identity, register_extension, registered_extensions
from e3nn_mlx.ops_tp import tensor_product
from e3nn_mlx.profiling import benchmark_callable, compare_eager_and_compiled


@pytest.mark.mlx
def test_compile_or_identity_matches_eager_for_arrays() -> None:
    mx = mlx_backend._require()

    def fn(x):
        return x * 2.0 + 1.0

    compiled = compile_or_identity(fn)
    array = mlx_backend.asarray([1.0, 2.0, 3.0])
    eager = fn(array)
    compiled_out = compiled(array)
    mx.eval(compiled_out)
    assert eager.tolist() == compiled_out.tolist()


@pytest.mark.mlx
def test_profiling_helpers_return_positive_timings() -> None:
    array = mlx_backend.asarray([[0.2, 0.3, 0.4]] * 32)
    eager_ms = benchmark_callable(lambda x: x * 2.0 + 1.0, array, iters=2)
    result = compare_eager_and_compiled("affine", lambda x: x * 2.0 + 1.0, array, iters=2)
    assert eager_ms >= 0.0
    assert result.eager_ms >= 0.0
    assert result.compiled_ms >= 0.0


@pytest.mark.mlx
def test_extension_seam_can_override_tensor_product() -> None:
    def fake_tp(plan, left, right, *, weights=None):
        return IrrepsArray("0e", mlx_backend.asarray([[123.0]]))

    register_extension(ExtensionKernel(name="fake_tp", fn=fake_tp, has_custom_vjp=True, note="test"))
    left = IrrepsArray("1o", mlx_backend.asarray([[1.0, 0.0, 0.0]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[0.0, 1.0, 0.0]]))
    out = tensor_product(left, right, extension="fake_tp")
    assert "fake_tp" in registered_extensions()
    assert out.array.tolist()[0] == [123.0]
