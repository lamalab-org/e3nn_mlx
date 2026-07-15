from __future__ import annotations

import pytest

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray


@pytest.mark.mlx
def test_irreps_array_chunking_is_deterministic() -> None:
    array = mlx_backend.asarray([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    irreps_array = IrrepsArray("0e + 2x1o", array)

    assert [str(chunk.irrep) for chunk in irreps_array.chunks] == ["0e", "1o"]
    assert [chunk.multiplicity for chunk in irreps_array.chunks] == [1, 2]
    assert [chunk.width for chunk in irreps_array.chunks] == [1, 6]

    chunk_arrays = irreps_array.chunk_arrays()
    assert chunk_arrays[0].shape == (1, 1)
    assert chunk_arrays[1].shape == (1, 6)


@pytest.mark.mlx
def test_irreps_array_from_chunks_round_trip() -> None:
    scalar = mlx_backend.asarray([[1.0]])
    vectors = mlx_backend.asarray([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    irreps_array = IrrepsArray.from_chunks("0e + 2x1o", [scalar, vectors], backend=mlx_backend)
    chunk_arrays = irreps_array.chunk_arrays()

    assert irreps_array.shape == (1, 7)
    assert chunk_arrays[0].tolist() == [[1.0]]
    assert chunk_arrays[1].tolist() == [[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]


@pytest.mark.mlx
def test_irreps_array_regroup_reorders_data_with_metadata() -> None:
    irreps_array = IrrepsArray(
        "1o+0e+2x1o",
        mlx_backend.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]),
    )
    regrouped = irreps_array.regroup()
    assert str(regrouped.irreps) == "0e+3x1o"
    assert regrouped.array.tolist() == [[4.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
