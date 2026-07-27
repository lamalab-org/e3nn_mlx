"""MLX implementations of the shared benchmark workloads."""

from __future__ import annotations

import importlib.metadata
from typing import Any, Callable

from .common import Task
from .workloads import CASE_DESCRIPTIONS, ring_edges, spherical_irreps


_USE_CUSTOM_KERNELS = True


def configure(*, use_custom_kernels: bool = True) -> None:
    global _USE_CUSTOM_KERNELS
    _USE_CUSTOM_KERNELS = bool(use_custom_kernels)


def _imports():
    import mlx.core as mx
    import mlx.nn as nn

    import e3nn_mlx as e3nn
    from e3nn_mlx.models.v2106 import Compose, Convolution, MessagePassing

    return mx, nn, e3nn, Compose, Convolution, MessagePassing


def backend_metadata() -> dict[str, Any]:
    mx, _, _, _, _, _ = _imports()
    return {
        "framework": "mlx",
        "framework_version": importlib.metadata.version("mlx"),
        "device": str(mx.default_device()),
        "compiled": True,
    }


def _sync(result) -> None:
    mx, _, _, _, _, _ = _imports()
    mx.eval(result)
    mx.synchronize()


def _task(
    *,
    name: str,
    config: dict[str, Any],
    item_count: int,
    forward: Callable[[], Any],
    compile_forward: Callable[[], Callable[[], Any]],
    train: Callable[[], Any] | None = None,
    compile_train: Callable[[], Callable[[], Any]] | None = None,
) -> Task:
    mx, _, _, _, _, _ = _imports()
    return Task(
        name=name,
        family=name,
        description=CASE_DESCRIPTIONS[name],
        config=config,
        item_count=item_count,
        forward=forward,
        synchronize=_sync,
        compile_forward=compile_forward,
        train=train,
        compile_train=compile_train,
        reset_peak_memory=mx.reset_peak_memory,
        peak_memory=mx.get_peak_memory,
    )


def _module_train_functions(module, loss, arguments):
    mx, nn, _, _, _, _ = _imports()
    value_grad = nn.value_and_grad(module, loss)

    def eager():
        return value_grad(*arguments)

    def compiler():
        compiled = mx.compile(value_grad)
        return lambda: compiled(*arguments)

    return eager, compiler


def _activate_alpha(module) -> None:
    mx, _, _, Compose, _, _ = _imports()
    layers = module.layers if hasattr(module, "layers") else module.mp.layers
    for layer in layers:
        convolution = layer.first if isinstance(layer, Compose) else layer
        convolution.alpha.update(
            {"weight": 0.05 * mx.random.normal(shape=convolution.alpha.weight.shape)}
        )


def _edge_data(config):
    mx, _, e3nn, _, _, _ = _imports()
    source, destination = ring_edges(config["nodes"], config["neighbors"])
    edge_src = mx.array(source, dtype=mx.int32)
    edge_dst = mx.array(destination, dtype=mx.int32)
    edge_vectors = mx.random.normal(shape=(len(source), 3))
    irreps_edge = e3nn.Irreps.spherical_harmonics(config["lmax"])
    edge_attr = e3nn.spherical_harmonics(
        irreps_edge,
        edge_vectors,
        normalize=True,
        normalization="component",
        use_custom_kernel=_USE_CUSTOM_KERNELS,
    )
    return edge_src, edge_dst, edge_vectors, irreps_edge, edge_attr


def build_spherical_harmonics(config: dict[str, Any]) -> Task:
    mx, _, e3nn, _, _, _ = _imports()
    vectors = mx.random.normal(shape=(config["items"], 3))
    degrees = list(range(config["lmax"] + 1))

    def raw(value):
        return e3nn.spherical_harmonics(
            degrees,
            value,
            normalize=True,
            normalization="component",
            use_custom_kernel=_USE_CUSTOM_KERNELS,
        )

    def loss(value):
        return mx.mean(raw(value) ** 2)

    value_grad = mx.value_and_grad(loss)
    return _task(
        name="spherical_harmonics",
        config=config,
        item_count=config["items"],
        forward=lambda: raw(vectors),
        compile_forward=lambda: (lambda compiled=mx.compile(raw): compiled(vectors)),
        train=lambda: value_grad(vectors),
        compile_train=lambda: (
            lambda compiled=mx.compile(value_grad): compiled(vectors)
        ),
    )


def build_full_tensor_product(config: dict[str, Any]) -> Task:
    mx, _, e3nn, _, _, _ = _imports()
    irreps = spherical_irreps(config["mul"], config["lmax"])
    module = e3nn.FullTensorProduct(
        irreps, irreps, use_custom_kernel=_USE_CUSTOM_KERNELS
    )
    left = mx.random.normal(shape=(config["items"], module.irreps_in1.dim))
    right = mx.random.normal(shape=(config["items"], module.irreps_in2.dim))

    def raw(first, second):
        return module(
            e3nn.IrrepsArray(module.irreps_in1, first),
            e3nn.IrrepsArray(module.irreps_in2, second),
        ).array

    def loss(first, second):
        return mx.mean(raw(first, second) ** 2)

    value_grad = mx.value_and_grad(loss, argnums=(0, 1))

    def compile_forward():
        compiled = mx.compile(raw)
        return lambda: compiled(left, right)

    def compile_train():
        compiled = mx.compile(value_grad)
        return lambda: compiled(left, right)

    return _task(
        name="full_tensor_product",
        config=config,
        item_count=config["items"],
        forward=lambda: raw(left, right),
        compile_forward=compile_forward,
        train=lambda: value_grad(left, right),
        compile_train=compile_train,
    )


def build_fully_connected_tensor_product(config: dict[str, Any]) -> Task:
    mx, _, e3nn, _, _, _ = _imports()
    irreps = spherical_irreps(config["mul"], config["lmax"])
    module = e3nn.FullyConnectedTensorProduct(
        irreps, irreps, irreps, use_custom_kernel=_USE_CUSTOM_KERNELS
    )
    left = mx.random.normal(shape=(config["items"], module.irreps_in1.dim))
    right = mx.random.normal(shape=(config["items"], module.irreps_in2.dim))

    def raw(first, second):
        return module(
            e3nn.IrrepsArray(module.irreps_in1, first),
            e3nn.IrrepsArray(module.irreps_in2, second),
        ).array

    train, compile_train = _module_train_functions(
        module, lambda first, second: mx.mean(raw(first, second) ** 2), (left, right)
    )

    def compile_forward():
        compiled = mx.compile(raw)
        return lambda: compiled(left, right)

    return _task(
        name="fully_connected_tensor_product",
        config=config,
        item_count=config["items"],
        forward=lambda: raw(left, right),
        compile_forward=compile_forward,
        train=train,
        compile_train=compile_train,
    )


def build_linear(config: dict[str, Any]) -> Task:
    mx, _, e3nn, _, _, _ = _imports()
    irreps = spherical_irreps(config["mul"], config["lmax"])
    module = e3nn.Linear(irreps, irreps)
    values = mx.random.normal(shape=(config["items"], module.irreps_in.dim))

    def raw(array):
        return module(e3nn.IrrepsArray(module.irreps_in, array)).array

    train, compile_train = _module_train_functions(
        module, lambda array: mx.mean(raw(array) ** 2), (values,)
    )
    return _task(
        name="linear",
        config=config,
        item_count=config["items"],
        forward=lambda: raw(values),
        compile_forward=lambda: (lambda compiled=mx.compile(raw): compiled(values)),
        train=train,
        compile_train=compile_train,
    )


def _convolution_fixture(config):
    mx, _, e3nn, _, Convolution, _ = _imports()
    irreps_node = spherical_irreps(config["mul"], config["lmax"])
    edge_src, edge_dst, _, irreps_edge, edge_attr = _edge_data(config)
    radial = config.get("radial", 10)
    radial_hidden = config.get("radial_hidden", 64)
    module = Convolution(
        irreps_node,
        "0e",
        irreps_edge,
        irreps_node,
        [radial, radial_hidden],
        float(config["neighbors"]),
        use_custom_kernel=_USE_CUSTOM_KERNELS,
    )
    return mx, e3nn, module, edge_src, edge_dst, edge_attr


def build_weighted_tensor_product_uvu(config: dict[str, Any]) -> Task:
    mx, e3nn, convolution, edge_src, _, edge_attr = _convolution_fixture(config)
    module = convolution.tp
    left = mx.random.normal(shape=(edge_src.shape[0], module.irreps_in1.dim))
    right = edge_attr
    weights = mx.random.normal(shape=(edge_src.shape[0], module.weight_numel))

    def raw(first, second, weight):
        return module(
            e3nn.IrrepsArray(module.irreps_in1, first),
            e3nn.IrrepsArray(module.irreps_in2, second),
            weight,
        ).array

    value_grad = mx.value_and_grad(
        lambda first, second, weight: mx.mean(raw(first, second, weight) ** 2),
        argnums=(0, 1, 2),
    )
    arguments = left, right, weights
    return _task(
        name="weighted_tensor_product_uvu",
        config=config,
        item_count=edge_src.shape[0],
        forward=lambda: raw(*arguments),
        compile_forward=lambda: (
            lambda compiled=mx.compile(raw): compiled(*arguments)
        ),
        train=lambda: value_grad(*arguments),
        compile_train=lambda: (
            lambda compiled=mx.compile(value_grad): compiled(*arguments)
        ),
    )


def build_scatter_sum(config: dict[str, Any]) -> Task:
    mx, e3nn, convolution, edge_src, edge_dst, _ = _convolution_fixture(config)
    width = convolution.irreps_mid_execution.dim
    source = mx.random.normal(shape=(edge_src.shape[0], width))

    def raw(values):
        return e3nn.scatter_sum(
            values,
            edge_dst,
            config["nodes"],
            use_custom_kernel=_USE_CUSTOM_KERNELS,
        )

    value_grad = mx.value_and_grad(lambda values: mx.mean(raw(values) ** 2))
    return _task(
        name="scatter_sum",
        config=config,
        item_count=edge_src.shape[0],
        forward=lambda: raw(source),
        compile_forward=lambda: (lambda compiled=mx.compile(raw): compiled(source)),
        train=lambda: value_grad(source),
        compile_train=lambda: (
            lambda compiled=mx.compile(value_grad): compiled(source)
        ),
    )


def build_gate(config: dict[str, Any]) -> Task:
    mx, _, e3nn, _, _, _ = _imports()
    from e3nn_mlx.nn_gate import Gate

    scalars = f"{config['mul']}x0e"
    gated = " + ".join(
        f"{config['mul']}x{degree}{'e' if degree % 2 == 0 else 'o'}"
        for degree in range(1, config["lmax"] + 1)
    )
    gates = f"{config['mul'] * config['lmax']}x0e"
    module = Gate(
        scalars,
        [lambda value: value * mx.sigmoid(value)],
        gates,
        [mx.sigmoid],
        gated,
    )
    values = mx.random.normal(shape=(config["items"], module.irreps_in.dim))

    def raw(array):
        return module(e3nn.IrrepsArray(module.irreps_in, array)).array

    value_grad = mx.value_and_grad(lambda array: mx.mean(raw(array) ** 2))
    return _task(
        name="gate",
        config=config,
        item_count=config["items"],
        forward=lambda: raw(values),
        compile_forward=lambda: (lambda compiled=mx.compile(raw): compiled(values)),
        train=lambda: value_grad(values),
        compile_train=lambda: (
            lambda compiled=mx.compile(value_grad): compiled(values)
        ),
    )


def build_radial_mlp(config: dict[str, Any]) -> Task:
    mx, _, convolution, edge_src, _, _ = _convolution_fixture(config)
    module = convolution.fc
    values = mx.random.normal(shape=(edge_src.shape[0], config["radial"]))
    train, compile_train = _module_train_functions(
        module, lambda array: mx.mean(module(array) ** 2), (values,)
    )
    return _task(
        name="radial_mlp",
        config=config,
        item_count=edge_src.shape[0],
        forward=lambda: module(values),
        compile_forward=lambda: (
            lambda compiled=mx.compile(module): compiled(values)
        ),
        train=train,
        compile_train=compile_train,
    )


def build_v2106_convolution(config: dict[str, Any]) -> Task:
    mx, _, e3nn, _, Convolution, _ = _imports()
    irreps_node = spherical_irreps(config["mul"], config["lmax"])
    edge_src, edge_dst, _, irreps_edge, edge_attr = _edge_data(config)
    module = Convolution(
        irreps_node,
        "0e",
        irreps_edge,
        irreps_node,
        [config["radial"], config["radial_hidden"]],
        float(config["neighbors"]),
        use_custom_kernel=_USE_CUSTOM_KERNELS,
    )
    module.alpha.update({"weight": 0.05 * mx.random.normal(shape=module.alpha.weight.shape)})
    node_input = mx.random.normal(shape=(config["nodes"], module.irreps_node_input.dim))
    node_attr = mx.ones((config["nodes"], 1))
    edge_scalars = mx.random.normal(shape=(edge_src.shape[0], config["radial"]))
    arguments = node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars
    raw = module.forward_arrays
    train, compile_train = _module_train_functions(
        module, lambda *args: mx.mean(raw(*args) ** 2), arguments
    )

    def compile_forward():
        compiled = mx.compile(raw)
        return lambda: compiled(*arguments)

    return _task(
        name="v2106_convolution",
        config=config,
        item_count=edge_src.shape[0],
        forward=lambda: raw(*arguments),
        compile_forward=compile_forward,
        train=train,
        compile_train=compile_train,
    )


def build_v2106_message_passing(config: dict[str, Any]) -> Task:
    mx, _, _, _, _, MessagePassing = _imports()
    irreps_node = spherical_irreps(config["mul"], config["lmax"])
    edge_src, edge_dst, _, irreps_edge, edge_attr = _edge_data(config)
    module = MessagePassing(
        [irreps_node] * (config["layers"] + 2),
        "0e",
        irreps_edge,
        [config["radial"], config["radial_hidden"]],
        float(config["neighbors"]),
        use_custom_kernel=_USE_CUSTOM_KERNELS,
    )
    _activate_alpha(module)
    node_input = mx.random.normal(shape=(config["nodes"], module.irreps_node_input.dim))
    node_attr = mx.ones((config["nodes"], 1))
    edge_scalars = mx.random.normal(shape=(edge_src.shape[0], config["radial"]))
    arguments = node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars
    raw = module.forward_arrays
    train, compile_train = _module_train_functions(
        module, lambda *args: mx.mean(raw(*args) ** 2), arguments
    )

    def compile_forward():
        compiled = mx.compile(raw)
        return lambda: compiled(*arguments)

    return _task(
        name="v2106_message_passing",
        config=config,
        item_count=edge_src.shape[0] * len(module.layers),
        forward=lambda: raw(*arguments),
        compile_forward=compile_forward,
        train=train,
        compile_train=compile_train,
    )


def build_v2106_network(config: dict[str, Any]) -> Task:
    mx, _, e3nn, _, _, _ = _imports()
    irreps_node = spherical_irreps(config["mul"], config["lmax"])
    edge_src, edge_dst, _, _, _ = _edge_data(config)
    module = e3nn.NetworkForAGraphWithAttributes(
        irreps_node,
        "0e",
        "0e",
        irreps_node,
        max_radius=4.0,
        num_neighbors=float(config["neighbors"]),
        num_nodes=float(config["nodes"]),
        mul=config["mul"],
        layers=config["layers"],
        lmax=config["lmax"],
        pool_nodes=True,
        use_custom_kernel=_USE_CUSTOM_KERNELS,
    )
    _activate_alpha(module)
    positions = mx.random.normal(shape=(config["nodes"], 3))
    node_input = mx.random.normal(shape=(config["nodes"], module.irreps_node_input.dim))
    node_attr = mx.ones((config["nodes"], 1))
    edge_attr = mx.ones((edge_src.shape[0], 1))
    batch = mx.zeros((config["nodes"],), dtype=mx.int32)
    arguments = positions, node_input, node_attr, edge_attr, batch, edge_src, edge_dst

    def raw(pos, features, attributes, edge_attributes, graph_batch, source, destination):
        return module.forward_with_edges(
            pos,
            features,
            attributes,
            edge_attributes,
            graph_batch,
            source,
            destination,
            num_graphs=1,
        ).array

    train, compile_train = _module_train_functions(
        module, lambda *args: mx.mean(raw(*args) ** 2), arguments
    )

    def compile_forward():
        compiled = mx.compile(raw)
        return lambda: compiled(*arguments)

    return _task(
        name="v2106_network",
        config=config,
        item_count=edge_src.shape[0] * len(module.mp.layers),
        forward=lambda: raw(*arguments),
        compile_forward=compile_forward,
        train=train,
        compile_train=compile_train,
    )


BUILDERS = {
    "spherical_harmonics": build_spherical_harmonics,
    "full_tensor_product": build_full_tensor_product,
    "fully_connected_tensor_product": build_fully_connected_tensor_product,
    "linear": build_linear,
    "weighted_tensor_product_uvu": build_weighted_tensor_product_uvu,
    "scatter_sum": build_scatter_sum,
    "gate": build_gate,
    "radial_mlp": build_radial_mlp,
    "v2106_convolution": build_v2106_convolution,
    "v2106_message_passing": build_v2106_message_passing,
    "v2106_network": build_v2106_network,
}


def build_tasks(workloads: dict[str, dict[str, Any]]) -> list[Task]:
    return [BUILDERS[name](config) for name, config in workloads.items()]
