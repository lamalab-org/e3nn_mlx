"""Small dependency-free graph operations for MLX point-cloud models."""

from __future__ import annotations

from .compat import mlx_metal_available, require_mlx


def scatter_sum(
    source,
    index,
    dim_size: int | None = None,
    *,
    use_custom_kernel: bool = False,
    jvp_safe: bool = False,
):
    """Sum rows of ``source`` into rows selected by one-dimensional ``index``.

    Set ``jvp_safe=True`` for forward-mode differentiation with respect to
    ``source``.  This path requires eager, fixed indices and implements the
    same reduction with a sorted prefix sum because MLX 0.31's indexed-add
    primitive has no JVP rule.
    """

    mx, _ = require_mlx()
    if source.ndim < 1:
        raise ValueError("source must have at least one dimension")
    if index.ndim != 1 or index.shape[0] != source.shape[0]:
        raise ValueError("index must be one-dimensional with one entry per source row")
    if not mx.issubdtype(index.dtype, mx.integer):
        raise TypeError("index must have an integer dtype")
    if dim_size is None:
        dim_size = 0 if index.size == 0 else int(mx.max(index).item()) + 1
    if dim_size < 0:
        raise ValueError("dim_size must be non-negative")
    if jvp_safe:
        if use_custom_kernel:
            raise ValueError("jvp_safe=True requires use_custom_kernel=False")
        return _fixed_index_scatter_sum(source, index, dim_size)
    use_metal = (
        use_custom_kernel
        and mlx_metal_available()
        and source.ndim >= 2
        and source.dtype == mx.float32
        and source.shape[0] > 0
    )
    if use_metal:
        try:
            minimum = int(mx.min(index).item())
            maximum = int(mx.max(index).item())
        except ValueError:
            # Dynamic indices cannot be inspected while tracing a transform;
            # retain the transformable general MLX scatter in that case.
            use_metal = False
        else:
            if minimum < 0 or maximum >= dim_size:
                raise ValueError("index values must satisfy 0 <= index < dim_size")
    if use_metal:
        from ._metal_scatter import make_operation

        return make_operation(index, dim_size, source.shape)(source)
    output = mx.zeros((dim_size, *source.shape[1:]), dtype=source.dtype)
    return output.at[index].add(source)


def _fixed_index_scatter_sum(source, index, dim_size: int):
    """Sparse JVP-safe scatter for an eager, fixed index array."""

    mx, _ = require_mlx()
    import numpy as np

    try:
        host_index = np.asarray(index)
    except Exception as error:
        raise RuntimeError(
            "jvp_safe scatter requires an eager fixed index array"
        ) from error
    if host_index.size:
        minimum = int(host_index.min())
        maximum = int(host_index.max())
        if minimum < 0 or maximum >= dim_size:
            raise ValueError("index values must satisfy 0 <= index < dim_size")

    order_host = np.argsort(host_index, kind="stable")
    sorted_index = host_index[order_host]
    rows = np.arange(dim_size, dtype=host_index.dtype)
    starts_host = np.searchsorted(sorted_index, rows, side="left")
    ends_host = np.searchsorted(sorted_index, rows, side="right")
    order = mx.array(order_host, dtype=mx.int32)
    starts = mx.array(starts_host, dtype=mx.int32)
    ends = mx.array(ends_host, dtype=mx.int32)

    sorted_source = source[order]
    prefix = mx.concatenate(
        (
            mx.zeros((1, *source.shape[1:]), dtype=source.dtype),
            mx.cumsum(sorted_source, axis=0),
        ),
        axis=0,
    )
    return prefix[ends] - prefix[starts]


def radius_graph(positions, radius: float, batch=None):
    """Build the directed, loop-free batched radius graph used by gate-points models.

    Topology construction is intentionally eager: MLX 0.31 has no device-side
    dynamic ``nonzero`` operation. Gradients should flow through edge vectors,
    not through the discrete neighbor selection itself.
    """

    mx, _ = require_mlx()
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (nodes, 3)")
    if radius <= 0:
        raise ValueError("radius must be positive")
    node_count = positions.shape[0]
    if batch is None:
        batch = mx.zeros((node_count,), dtype=mx.int32)
    if batch.ndim != 1 or batch.shape[0] != node_count:
        raise ValueError("batch must have shape (nodes,)")
    if not mx.issubdtype(batch.dtype, mx.integer):
        raise TypeError("batch must have an integer dtype")

    differences = positions[:, None, :] - positions[None, :, :]
    squared_distance = mx.sum(differences * differences, axis=-1)
    adjacency = (
        (squared_distance < radius * radius)
        & (squared_distance > 0)
        & (batch[:, None] == batch[None, :])
    )
    import numpy as np

    pairs = np.argwhere(np.asarray(adjacency))
    if pairs.size == 0:
        return mx.zeros((2, 0), dtype=mx.int32)
    return mx.array(pairs.T, dtype=mx.int32)
