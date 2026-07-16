"""Equivariant batch and instance normalization."""

from __future__ import annotations

from e3nn_core.irreps import Irreps

from .compat import mlx_module_base, require_mlx
from .irreps_array import IrrepsArray


class BatchNorm(mlx_module_base()):
    def __init__(
        self,
        irreps: Irreps | str,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        reduce: str = "mean",
        instance: bool = False,
        include_bias: bool = True,
        normalization: str = "component",
    ) -> None:
        super().__init__()
        if reduce not in ("mean", "max"):
            raise ValueError("reduce must be 'mean' or 'max'")
        if normalization not in ("norm", "component"):
            raise ValueError("normalization must be 'norm' or 'component'")
        self.irreps = Irreps(irreps).remove_zero_multiplicities()
        self.irreps_in = self.irreps_out = self.irreps
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.reduce = reduce
        self.instance = bool(instance)
        self.include_bias = bool(include_bias)
        self.normalization = normalization
        num_scalar = sum(part.mul for part in self.irreps if part.ir.is_scalar())
        mx, _ = require_mlx()
        self._running_mean = None if instance else mx.zeros((num_scalar,))
        self._running_var = None if instance else mx.ones((self.irreps.num_irreps,))
        self.weight = mx.ones((self.irreps.num_irreps,)) if affine else None
        self.bias = mx.zeros((num_scalar,)) if affine and include_bias and num_scalar else None

    @property
    def running_mean(self):
        return self._running_mean

    @property
    def running_var(self):
        return self._running_var

    def __repr__(self) -> str:
        return f"BatchNorm ({self.irreps}, eps={self.eps}, momentum={self.momentum})"

    def __call__(self, features: IrrepsArray) -> IrrepsArray:
        if features.irreps != self.irreps:
            raise ValueError("input irreps do not match BatchNorm.irreps")
        if not features.leading_shape:
            raise ValueError("BatchNorm requires a batch dimension")
        mx, _ = require_mlx()
        batch = features.leading_shape[0]
        sample_count = 1
        for size in features.leading_shape[1:]:
            sample_count *= size
        reshaped = features.array.reshape(batch, sample_count, self.irreps.dim)
        output_chunks = []
        new_means = []
        new_vars = []
        scalar_cursor = feature_cursor = 0
        for part, chunk in zip(self.irreps, IrrepsArray(self.irreps, reshaped).chunk_arrays(), strict=True):
            field = chunk.reshape(batch, sample_count, part.mul, part.ir.dim)
            if part.ir.is_scalar():
                if self.training or self.instance:
                    mean = mx.mean(field, axis=1).reshape(batch, part.mul) if self.instance else mx.mean(field, axis=(0, 1)).reshape(part.mul)
                    if not self.instance:
                        running = self._running_mean[scalar_cursor : scalar_cursor + part.mul]
                        new_means.append((1.0 - self.momentum) * running + self.momentum * mx.stop_gradient(mean))
                else:
                    mean = self._running_mean[scalar_cursor : scalar_cursor + part.mul]
                field = field - mean.reshape(batch if self.instance else 1, 1, part.mul, 1)
                scalar_cursor += part.mul

            if self.training or self.instance:
                squared = mx.sum(field * field, axis=-1) if self.normalization == "norm" else mx.mean(field * field, axis=-1)
                reduced = mx.mean(squared, axis=1) if self.reduce == "mean" else mx.max(squared, axis=1)
                if self.instance:
                    variance = reduced
                else:
                    variance = mx.mean(reduced, axis=0)
                    running = self._running_var[feature_cursor : feature_cursor + part.mul]
                    new_vars.append((1.0 - self.momentum) * running + self.momentum * mx.stop_gradient(variance))
            else:
                variance = self._running_var[feature_cursor : feature_cursor + part.mul]
            scale = mx.rsqrt(variance + self.eps)
            if self.affine:
                scale = scale * self.weight[feature_cursor : feature_cursor + part.mul]
            field = field * scale.reshape(batch if self.instance else 1, 1, part.mul, 1)
            if self.bias is not None and part.ir.is_scalar():
                start = scalar_cursor - part.mul
                field = field + self.bias[start : start + part.mul].reshape(1, 1, part.mul, 1)
            output_chunks.append(field.reshape(batch, sample_count, part.dim))
            feature_cursor += part.mul

        if self.training and not self.instance:
            if new_means:
                self._running_mean = mx.concatenate(new_means)
            if new_vars:
                self._running_var = mx.concatenate(new_vars)
        output = mx.concatenate(output_chunks, axis=-1) if output_chunks else mx.zeros_like(reshaped)
        return IrrepsArray(self.irreps, output.reshape(*features.leading_shape, self.irreps.dim))
