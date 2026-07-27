"""MLX implementations for the compact three-way benchmark."""

from __future__ import annotations

import importlib.metadata
from typing import Any, Callable

from .common import Task
from .workloads import CASE_DESCRIPTIONS, spherical_irreps


_USE_CUSTOM_KERNELS = False


def configure(*, use_custom_kernels: bool) -> None:
    global _USE_CUSTOM_KERNELS
    _USE_CUSTOM_KERNELS = bool(use_custom_kernels)


def _imports():
    import mlx.core as mx
    import mlx.nn as nn

    from e3nn_mlx import o3, scatter_sum

    return mx, nn, o3, scatter_sum


def backend_metadata() -> dict[str, Any]:
    mx, _, _, _ = _imports()
    return {
        "framework": "mlx",
        "framework_version": importlib.metadata.version("mlx"),
        "device": str(mx.default_device()),
        "compiled": True,
        "custom_kernels": _USE_CUSTOM_KERNELS,
    }


def _sync(result) -> None:
    mx, _, _, _ = _imports()
    mx.eval(result)
    mx.synchronize()


def _task(
    *,
    name: str,
    config: dict[str, Any],
    item_count: int,
    forward: Callable[[], Any],
    compile_forward: Callable[[], Callable[[], Any]],
    train: Callable[[], Any],
    compile_train: Callable[[], Callable[[], Any]],
    dispatch: str = "general-mlx",
) -> Task:
    mx, _, _, _ = _imports()
    return Task(
        name=name,
        family=name,
        description=CASE_DESCRIPTIONS[name],
        config=config,
        item_count=item_count,
        forward=forward,
        synchronize=_sync,
        dispatch=dispatch,
        compile_forward=compile_forward,
        train=train,
        compile_train=compile_train,
        reset_peak_memory=mx.reset_peak_memory,
        peak_memory=mx.get_peak_memory,
    )


def _tensor_product_dispatch(module, left, right, weight=None) -> str:
    if not _USE_CUSTOM_KERNELS:
        return "general-mlx"
    kind = module._metal_dispatch_kind(left, right, weight)
    if kind is None:
        return "general-mlx (kernel fallback)"
    return f"metal-{kind.replace('_', '-')}"


def _compiled(function, *arguments):
    mx, _, _, _ = _imports()

    def prepare():
        compiled = mx.compile(function)
        return lambda: compiled(*arguments)

    return prepare


def _module_grad(module, loss, *arguments):
    mx, nn, _, _ = _imports()
    value_grad = nn.value_and_grad(module, loss)
    return (
        lambda: value_grad(*arguments),
        _compiled(value_grad, *arguments),
    )


def build_spherical_harmonics(config: dict[str, Any]) -> Task:
    mx, _, o3, _ = _imports()
    vectors = mx.random.normal((config["items"], 3))
    degrees = list(range(config["lmax"] + 1))

    def forward(value):
        return o3.spherical_harmonics(
            degrees,
            value,
            normalize=True,
            normalization="component",
            use_custom_kernel=_USE_CUSTOM_KERNELS,
        )

    value_grad = mx.value_and_grad(lambda value: mx.mean(forward(value) ** 2))
    return _task(
        name="spherical_harmonics",
        config=config,
        item_count=config["items"],
        forward=lambda: forward(vectors),
        compile_forward=_compiled(forward, vectors),
        train=lambda: value_grad(vectors),
        compile_train=_compiled(value_grad, vectors),
        dispatch=(
            "metal-spherical-harmonics"
            if _USE_CUSTOM_KERNELS and config["lmax"] <= 4
            else (
                "general-mlx (kernel fallback)"
                if _USE_CUSTOM_KERNELS
                else "general-mlx"
            )
        ),
    )


def build_full_tensor_product(config: dict[str, Any]) -> Task:
    mx, _, o3, _ = _imports()
    irreps = spherical_irreps(config["mul"], config["lmax"])
    module = o3.FullTensorProduct(
        irreps,
        irreps,
        use_custom_kernel=_USE_CUSTOM_KERNELS,
    )
    left = mx.random.normal((config["items"], module.irreps_in1.dim))
    right = mx.random.normal((config["items"], module.irreps_in2.dim))

    def forward(first, second):
        return module(first, second)

    value_grad = mx.value_and_grad(
        lambda first, second: mx.mean(forward(first, second) ** 2),
        argnums=(0, 1),
    )
    return _task(
        name="full_tensor_product",
        config=config,
        item_count=config["items"],
        forward=lambda: forward(left, right),
        compile_forward=_compiled(forward, left, right),
        train=lambda: value_grad(left, right),
        compile_train=_compiled(value_grad, left, right),
        dispatch=_tensor_product_dispatch(module, left, right),
    )


def build_fully_connected_tensor_product(config: dict[str, Any]) -> Task:
    mx, _, o3, _ = _imports()
    irreps = spherical_irreps(config["mul"], config["lmax"])
    module = o3.FullyConnectedTensorProduct(
        irreps,
        irreps,
        irreps,
        use_custom_kernel=_USE_CUSTOM_KERNELS,
    )
    left = mx.random.normal((config["items"], module.irreps_in1.dim))
    right = mx.random.normal((config["items"], module.irreps_in2.dim))

    def forward(first, second):
        return module(first, second)

    train, compile_train = _module_grad(
        module,
        lambda first, second: mx.mean(forward(first, second) ** 2),
        left,
        right,
    )
    return _task(
        name="fully_connected_tensor_product",
        config=config,
        item_count=config["items"],
        forward=lambda: forward(left, right),
        compile_forward=_compiled(forward, left, right),
        train=train,
        compile_train=compile_train,
        dispatch=_tensor_product_dispatch(module, left, right, module.weight),
    )


def _weighted_uvu(o3, config):
    irreps_left = o3.Irreps(spherical_irreps(config["mul"], config["lmax"]))
    irreps_right = o3.Irreps.spherical_harmonics(config["lmax"])
    outputs = []
    instructions = []
    for i_left, left in enumerate(irreps_left):
        for i_right, right in enumerate(irreps_right):
            for out in irreps_left:
                if out.ir not in (left.ir * right.ir):
                    continue
                outputs.append(str(out))
                instructions.append(
                    (i_left, i_right, len(outputs) - 1, "uvu", True)
                )
    return o3.TensorProduct(
        irreps_left,
        irreps_right,
        " + ".join(outputs),
        instructions,
        internal_weights=False,
        shared_weights=False,
        use_custom_kernel=_USE_CUSTOM_KERNELS,
    )


def build_weighted_tensor_product_uvu(config: dict[str, Any]) -> Task:
    mx, _, o3, _ = _imports()
    module = _weighted_uvu(o3, config)
    left = mx.random.normal((config["items"], module.irreps_in1.dim))
    right = mx.random.normal((config["items"], module.irreps_in2.dim))
    weights = mx.random.normal((config["items"], module.weight_numel))

    def forward(first, second, weight):
        return module(first, second, weight)

    value_grad = mx.value_and_grad(
        lambda first, second, weight: mx.mean(
            forward(first, second, weight) ** 2
        ),
        argnums=(0, 1, 2),
    )
    arguments = left, right, weights
    return _task(
        name="weighted_tensor_product_uvu",
        config=config,
        item_count=config["items"],
        forward=lambda: forward(*arguments),
        compile_forward=_compiled(forward, *arguments),
        train=lambda: value_grad(*arguments),
        compile_train=_compiled(value_grad, *arguments),
        dispatch=_tensor_product_dispatch(module, *arguments),
    )


def build_linear(config: dict[str, Any]) -> Task:
    mx, _, o3, _ = _imports()
    irreps = spherical_irreps(config["mul"], config["lmax"])
    module = o3.Linear(irreps, irreps)
    values = mx.random.normal((config["items"], module.irreps_in.dim))

    def forward(value):
        return module(value)

    train, compile_train = _module_grad(
        module,
        lambda value: mx.mean(forward(value) ** 2),
        values,
    )
    return _task(
        name="linear",
        config=config,
        item_count=config["items"],
        forward=lambda: forward(values),
        compile_forward=_compiled(forward, values),
        train=train,
        compile_train=compile_train,
        dispatch="general-mlx",
    )


def build_scatter_sum(config: dict[str, Any]) -> Task:
    mx, _, _, scatter_sum = _imports()
    values = mx.random.normal((config["items"], config["width"]))
    index = mx.arange(config["items"], dtype=mx.int32) % config["nodes"]

    def forward(source):
        return scatter_sum(
            source,
            index,
            config["nodes"],
            use_custom_kernel=_USE_CUSTOM_KERNELS,
        )

    value_grad = mx.value_and_grad(lambda source: mx.mean(forward(source) ** 2))
    return _task(
        name="scatter_sum",
        config=config,
        item_count=config["items"],
        forward=lambda: forward(values),
        compile_forward=_compiled(forward, values),
        train=lambda: value_grad(values),
        compile_train=_compiled(value_grad, values),
        dispatch=(
            "metal-scatter-sum" if _USE_CUSTOM_KERNELS else "general-mlx"
        ),
    )


BUILDERS = {
    "spherical_harmonics": build_spherical_harmonics,
    "full_tensor_product": build_full_tensor_product,
    "fully_connected_tensor_product": build_fully_connected_tensor_product,
    "weighted_tensor_product_uvu": build_weighted_tensor_product_uvu,
    "linear": build_linear,
    "scatter_sum": build_scatter_sum,
}
