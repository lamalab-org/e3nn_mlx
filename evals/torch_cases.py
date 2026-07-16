"""PyTorch/e3nn implementations of the shared benchmark workloads."""

from __future__ import annotations

import importlib.metadata
from typing import Any, Callable

from .common import Task
from .workloads import CASE_DESCRIPTIONS, ring_edges, spherical_irreps


_DEVICE = "cpu"
_ENABLE_COMPILE = False


def configure(*, device: str, enable_compile: bool) -> None:
    global _DEVICE, _ENABLE_COMPILE
    import torch

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available in this interpreter")
    _DEVICE = device
    _ENABLE_COMPILE = enable_compile
    torch.manual_seed(0)


def _imports():
    import torch
    from e3nn import o3
    from e3nn.nn.models.v2106.gate_points_message_passing import MessagePassing
    from e3nn.nn.models.v2106.gate_points_networks import (
        NetworkForAGraphWithAttributes,
    )
    from e3nn.nn.models.v2106.points_convolution import Convolution

    return torch, o3, Convolution, MessagePassing, NetworkForAGraphWithAttributes


def backend_metadata() -> dict[str, Any]:
    torch, _, _, _, _ = _imports()
    return {
        "framework": "torch",
        "framework_version": importlib.metadata.version("torch"),
        "e3nn_version": importlib.metadata.version("e3nn"),
        "device": _DEVICE,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "compiled": _ENABLE_COMPILE,
    }


def _sync(_result) -> None:
    torch, _, _, _, _ = _imports()
    if _DEVICE == "mps":
        torch.mps.synchronize()


def _inference(function: Callable[[], Any]) -> Callable[[], Any]:
    torch, _, _, _, _ = _imports()
    return torch.inference_mode()(function)


def _compiler(function: Callable[[], Any]) -> Callable[[], Callable[[], Any]] | None:
    if not _ENABLE_COMPILE:
        return None
    torch, _, _, _, _ = _imports()
    return lambda: torch.compile(function)


def _task(
    *,
    name: str,
    config: dict[str, Any],
    item_count: int,
    forward: Callable[[], Any],
    train: Callable[[], Any] | None,
) -> Task:
    return Task(
        name=name,
        family=name,
        description=CASE_DESCRIPTIONS[name],
        config=config,
        item_count=item_count,
        forward=_inference(forward),
        synchronize=_sync,
        compile_forward=_compiler(_inference(forward)),
        train=train,
        compile_train=_compiler(train) if train is not None else None,
    )


def _module_train(module, forward: Callable[[], Any]):
    def train():
        for parameter in module.parameters():
            parameter.grad = None
        output = forward()
        loss = output.square().mean()
        loss.backward()
        return loss

    return train


def _activate_alpha(module) -> None:
    torch, _, _, _, _ = _imports()
    layers = module.layers if hasattr(module, "layers") else module.mp.layers
    with torch.no_grad():
        for layer in layers:
            convolution = layer.first if hasattr(layer, "first") else layer
            convolution.alpha.weight.normal_(mean=0.0, std=0.05)


def _edge_data(config):
    torch, o3, _, _, _ = _imports()
    source, destination = ring_edges(config["nodes"], config["neighbors"])
    edge_src = torch.tensor(source, dtype=torch.long, device=_DEVICE)
    edge_dst = torch.tensor(destination, dtype=torch.long, device=_DEVICE)
    edge_vectors = torch.randn(len(source), 3, device=_DEVICE)
    irreps_edge = o3.Irreps.spherical_harmonics(config["lmax"])
    edge_attr = o3.spherical_harmonics(
        irreps_edge,
        edge_vectors,
        normalize=True,
        normalization="component",
    )
    return edge_src, edge_dst, edge_vectors, irreps_edge, edge_attr


def build_spherical_harmonics(config: dict[str, Any]) -> Task:
    torch, o3, _, _, _ = _imports()
    vectors = torch.randn(config["items"], 3, device=_DEVICE, requires_grad=True)
    degrees = list(range(config["lmax"] + 1))

    def raw():
        return o3.spherical_harmonics(
            degrees,
            vectors,
            normalize=True,
            normalization="component",
        )

    def train():
        vectors.grad = None
        loss = raw().square().mean()
        loss.backward()
        return loss

    return _task(
        name="spherical_harmonics",
        config=config,
        item_count=config["items"],
        forward=raw,
        train=train,
    )


def build_full_tensor_product(config: dict[str, Any]) -> Task:
    torch, o3, _, _, _ = _imports()
    irreps = o3.Irreps(spherical_irreps(config["mul"], config["lmax"]))
    module = o3.FullTensorProduct(irreps, irreps).to(_DEVICE)
    left = torch.randn(config["items"], irreps.dim, device=_DEVICE, requires_grad=True)
    right = torch.randn(config["items"], irreps.dim, device=_DEVICE, requires_grad=True)

    def raw():
        return module(left, right)

    def train():
        left.grad = None
        right.grad = None
        loss = raw().square().mean()
        loss.backward()
        return loss

    return _task(
        name="full_tensor_product",
        config=config,
        item_count=config["items"],
        forward=raw,
        train=train,
    )


def build_fully_connected_tensor_product(config: dict[str, Any]) -> Task:
    torch, o3, _, _, _ = _imports()
    irreps = o3.Irreps(spherical_irreps(config["mul"], config["lmax"]))
    module = o3.FullyConnectedTensorProduct(irreps, irreps, irreps).to(_DEVICE)
    left = torch.randn(config["items"], irreps.dim, device=_DEVICE)
    right = torch.randn(config["items"], irreps.dim, device=_DEVICE)
    raw = lambda: module(left, right)
    return _task(
        name="fully_connected_tensor_product",
        config=config,
        item_count=config["items"],
        forward=raw,
        train=_module_train(module, raw),
    )


def build_linear(config: dict[str, Any]) -> Task:
    torch, o3, _, _, _ = _imports()
    irreps = o3.Irreps(spherical_irreps(config["mul"], config["lmax"]))
    module = o3.Linear(irreps, irreps).to(_DEVICE)
    values = torch.randn(config["items"], irreps.dim, device=_DEVICE)
    raw = lambda: module(values)
    return _task(
        name="linear",
        config=config,
        item_count=config["items"],
        forward=raw,
        train=_module_train(module, raw),
    )


def build_v2106_convolution(config: dict[str, Any]) -> Task:
    torch, o3, Convolution, _, _ = _imports()
    irreps_node = o3.Irreps(spherical_irreps(config["mul"], config["lmax"]))
    edge_src, edge_dst, _, irreps_edge, edge_attr = _edge_data(config)
    module = Convolution(
        irreps_node,
        "0e",
        irreps_edge,
        irreps_node,
        [config["radial"], config["radial_hidden"]],
        float(config["neighbors"]),
    ).to(_DEVICE)
    with torch.no_grad():
        module.alpha.weight.normal_(mean=0.0, std=0.05)
    node_input = torch.randn(config["nodes"], irreps_node.dim, device=_DEVICE)
    node_attr = torch.ones(config["nodes"], 1, device=_DEVICE)
    edge_scalars = torch.randn(edge_src.shape[0], config["radial"], device=_DEVICE)
    raw = lambda: module(
        node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars
    )
    return _task(
        name="v2106_convolution",
        config=config,
        item_count=edge_src.shape[0],
        forward=raw,
        train=_module_train(module, raw),
    )


def build_v2106_message_passing(config: dict[str, Any]) -> Task:
    torch, o3, _, MessagePassing, _ = _imports()
    irreps_node = o3.Irreps(spherical_irreps(config["mul"], config["lmax"]))
    edge_src, edge_dst, _, irreps_edge, edge_attr = _edge_data(config)
    module = MessagePassing(
        [irreps_node] * (config["layers"] + 2),
        "0e",
        irreps_edge,
        [config["radial"], config["radial_hidden"]],
        float(config["neighbors"]),
    ).to(_DEVICE)
    _activate_alpha(module)
    node_input = torch.randn(config["nodes"], irreps_node.dim, device=_DEVICE)
    node_attr = torch.ones(config["nodes"], 1, device=_DEVICE)
    edge_scalars = torch.randn(edge_src.shape[0], config["radial"], device=_DEVICE)
    raw = lambda: module(
        node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars
    )
    return _task(
        name="v2106_message_passing",
        config=config,
        item_count=edge_src.shape[0] * len(module.layers),
        forward=raw,
        train=_module_train(module, raw),
    )


def build_v2106_network(config: dict[str, Any]) -> Task:
    torch, o3, _, _, Network = _imports()
    irreps_node = o3.Irreps(spherical_irreps(config["mul"], config["lmax"]))
    edge_src, edge_dst, _, _, _ = _edge_data(config)
    module = Network(
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
    ).to(_DEVICE)
    _activate_alpha(module)
    positions = torch.randn(config["nodes"], 3, device=_DEVICE)
    node_input = torch.randn(config["nodes"], irreps_node.dim, device=_DEVICE)
    node_attr = torch.ones(config["nodes"], 1, device=_DEVICE)
    edge_attr = torch.ones(edge_src.shape[0], 1, device=_DEVICE)
    edge_index = torch.stack([edge_src, edge_dst])
    data = {
        "pos": positions,
        "node_input": node_input,
        "node_attr": node_attr,
        "edge_attr": edge_attr,
        "edge_index": edge_index,
    }
    raw = lambda: module(data)
    return _task(
        name="v2106_network",
        config=config,
        item_count=edge_src.shape[0] * len(module.mp.layers),
        forward=raw,
        train=_module_train(module, raw),
    )


BUILDERS = {
    "spherical_harmonics": build_spherical_harmonics,
    "full_tensor_product": build_full_tensor_product,
    "fully_connected_tensor_product": build_fully_connected_tensor_product,
    "linear": build_linear,
    "v2106_convolution": build_v2106_convolution,
    "v2106_message_passing": build_v2106_message_passing,
    "v2106_network": build_v2106_network,
}


def build_tasks(workloads: dict[str, dict[str, Any]]) -> list[Task]:
    return [BUILDERS[name](config) for name, config in workloads.items()]
