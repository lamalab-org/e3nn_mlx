"""Variance-normalized fully connected network."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .compat import mlx_module_base, require_mlx
from .nn_activation import _activation_properties


class _Layer(mlx_module_base()):
    def __init__(self, h_in: int, h_out: int, act, var_in: float, var_out: float) -> None:
        super().__init__()
        mx, _ = require_mlx()
        self.weight = mx.random.normal(shape=(h_in, h_out))
        self.h_in = h_in
        self.h_out = h_out
        self.act = act
        self.var_in = float(var_in)
        self.var_out = float(var_out)

    def __call__(self, values):
        if self.act is None:
            weight = self.weight / (self.h_in * self.var_in / self.var_out) ** 0.5
            return values @ weight
        weight = self.weight / (self.h_in * self.var_in) ** 0.5
        return self.var_out**0.5 * self.act(values @ weight)


class FullyConnectedNet(mlx_module_base()):
    def __init__(
        self,
        hs: Sequence[int],
        act: Callable[[Any], Any] | None = None,
        variance_in: float = 1.0,
        variance_out: float = 1.0,
        out_act: bool = False,
    ) -> None:
        super().__init__()
        self.hs = tuple(int(value) for value in hs)
        if len(self.hs) < 2 or any(value <= 0 for value in self.hs):
            raise ValueError("hs must contain at least two positive dimensions")
        normalized_act = None
        if act is not None:
            scale, _ = _activation_properties(act)
            normalized_act = lambda value: scale * act(value)
        var_in = float(variance_in)
        self._layer_names = []
        for index, (h_in, h_out) in enumerate(zip(self.hs, self.hs[1:])):
            is_last = index == len(self.hs) - 2
            var_out = float(variance_out) if is_last else 1.0
            layer_act = normalized_act if (not is_last or out_act) else None
            name = f"layer{index}"
            setattr(self, name, _Layer(h_in, h_out, layer_act, var_in, var_out))
            self._layer_names.append(name)
            var_in = var_out

    def __repr__(self) -> str:
        return f"FullyConnectedNet{list(self.hs)}"

    def __call__(self, values):
        if values.shape[-1] != self.hs[0]:
            raise ValueError(f"expected input dimension {self.hs[0]}")
        for name in self._layer_names:
            values = getattr(self, name)(values)
        return values
