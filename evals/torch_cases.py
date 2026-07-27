"""Torch CPU/e3nn implementations for the compact three-way benchmark."""

from __future__ import annotations

import importlib.metadata
import os
from typing import Any, Callable

from .common import Task
from .workloads import CASE_DESCRIPTIONS, spherical_irreps


_THREADS = max(1, os.cpu_count() or 1)


def configure() -> None:
    import torch

    torch.set_num_threads(_THREADS)
    try:
        torch.set_num_interop_threads(_THREADS)
    except RuntimeError:
        pass
    torch.manual_seed(0)


def _imports():
    import torch
    from e3nn import o3

    return torch, o3


def backend_metadata() -> dict[str, Any]:
    torch, _ = _imports()
    return {
        "framework": "torch",
        "framework_version": importlib.metadata.version("torch"),
        "e3nn_version": importlib.metadata.version("e3nn"),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "compiled": False,
    }


def _sync(_result) -> None:
    return None


def _task(
    *,
    name: str,
    config: dict[str, Any],
    item_count: int,
    forward: Callable[[], Any],
    train: Callable[[], Any],
) -> Task:
    torch, _ = _imports()
    return Task(
        name=name,
        family=name,
        description=CASE_DESCRIPTIONS[name],
        config=config,
        item_count=item_count,
        forward=torch.inference_mode()(forward),
        synchronize=_sync,
        dispatch="torch-eager",
        train=train,
    )


def _module_train(module, forward: Callable[[], Any]):
    def train():
        for parameter in module.parameters():
            parameter.grad = None
        loss = forward().square().mean()
        loss.backward()
        return loss

    return train


def build_spherical_harmonics(config: dict[str, Any]) -> Task:
    torch, o3 = _imports()
    vectors = torch.randn(config["items"], 3, requires_grad=True)
    degrees = list(range(config["lmax"] + 1))

    def forward():
        return o3.spherical_harmonics(
            degrees,
            vectors,
            normalize=True,
            normalization="component",
        )

    def train():
        vectors.grad = None
        loss = forward().square().mean()
        loss.backward()
        return loss

    return _task(
        name="spherical_harmonics",
        config=config,
        item_count=config["items"],
        forward=forward,
        train=train,
    )


def build_full_tensor_product(config: dict[str, Any]) -> Task:
    torch, o3 = _imports()
    irreps = o3.Irreps(spherical_irreps(config["mul"], config["lmax"]))
    module = o3.FullTensorProduct(irreps, irreps)
    left = torch.randn(config["items"], irreps.dim, requires_grad=True)
    right = torch.randn(config["items"], irreps.dim, requires_grad=True)

    def forward():
        return module(left, right)

    def train():
        left.grad = None
        right.grad = None
        loss = forward().square().mean()
        loss.backward()
        return loss

    return _task(
        name="full_tensor_product",
        config=config,
        item_count=config["items"],
        forward=forward,
        train=train,
    )


def build_fully_connected_tensor_product(config: dict[str, Any]) -> Task:
    torch, o3 = _imports()
    irreps = o3.Irreps(spherical_irreps(config["mul"], config["lmax"]))
    module = o3.FullyConnectedTensorProduct(irreps, irreps, irreps)
    left = torch.randn(config["items"], irreps.dim)
    right = torch.randn(config["items"], irreps.dim)
    forward = lambda: module(left, right)
    return _task(
        name="fully_connected_tensor_product",
        config=config,
        item_count=config["items"],
        forward=forward,
        train=_module_train(module, forward),
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
    )


def build_weighted_tensor_product_uvu(config: dict[str, Any]) -> Task:
    torch, o3 = _imports()
    module = _weighted_uvu(o3, config)
    left = torch.randn(
        config["items"], module.irreps_in1.dim, requires_grad=True
    )
    right = torch.randn(
        config["items"], module.irreps_in2.dim, requires_grad=True
    )
    weights = torch.randn(
        config["items"], module.weight_numel, requires_grad=True
    )

    def forward():
        return module(left, right, weights)

    def train():
        for value in (left, right, weights):
            value.grad = None
        loss = forward().square().mean()
        loss.backward()
        return loss

    return _task(
        name="weighted_tensor_product_uvu",
        config=config,
        item_count=config["items"],
        forward=forward,
        train=train,
    )


def build_linear(config: dict[str, Any]) -> Task:
    torch, o3 = _imports()
    irreps = o3.Irreps(spherical_irreps(config["mul"], config["lmax"]))
    module = o3.Linear(irreps, irreps)
    values = torch.randn(config["items"], irreps.dim)
    forward = lambda: module(values)
    return _task(
        name="linear",
        config=config,
        item_count=config["items"],
        forward=forward,
        train=_module_train(module, forward),
    )


def build_scatter_sum(config: dict[str, Any]) -> Task:
    torch, _ = _imports()
    values = torch.randn(
        config["items"],
        config["width"],
        requires_grad=True,
    )
    index = torch.arange(config["items"]) % config["nodes"]

    def forward():
        output = torch.zeros(config["nodes"], config["width"])
        return output.index_add(0, index, values)

    def train():
        values.grad = None
        loss = forward().square().mean()
        loss.backward()
        return loss

    return _task(
        name="scatter_sum",
        config=config,
        item_count=config["items"],
        forward=forward,
        train=train,
    )


BUILDERS = {
    "spherical_harmonics": build_spherical_harmonics,
    "full_tensor_product": build_full_tensor_product,
    "fully_connected_tensor_product": build_fully_connected_tensor_product,
    "weighted_tensor_product_uvu": build_weighted_tensor_product_uvu,
    "linear": build_linear,
    "scatter_sum": build_scatter_sum,
}
