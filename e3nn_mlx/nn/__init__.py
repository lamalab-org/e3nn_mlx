"""e3nn-compatible :mod:`nn` namespace backed by MLX.

Representation-aware modules accept raw MLX arrays in this namespace. Typed
``IrrepsArray`` inputs continue to produce typed outputs.
"""

from __future__ import annotations

from .._high_level import ensure_irreps_array, unwrap_irreps_arrays
from ..nn_activation import Activation as _Activation
from ..nn_batchnorm import BatchNorm as _BatchNorm
from ..nn_dropout import Dropout as _Dropout
from ..nn_extract import Extract as _Extract, ExtractIr as _ExtractIr
from ..nn_fc import FullyConnectedNet as _FullyConnectedNet
from ..nn_gate import Gate as _Gate
from ..nn_identity import Identity as _Identity
from ..nn_normact import NormActivation as _NormActivation
from ..nn_s2act import S2Activation as _S2Activation
from ..nn_so3act import SO3Activation as _SO3Activation


class _UnaryIrrepsIO:
    def __call__(self, features):
        typed, raw = ensure_irreps_array(features, self.irreps_in)
        return unwrap_irreps_arrays(super().__call__(typed), raw)

    def forward(self, features):
        return self(features)


class Activation(_UnaryIrrepsIO, _Activation):
    """Raw-array compatible Activation."""


class BatchNorm(_UnaryIrrepsIO, _BatchNorm):
    """Raw-array compatible BatchNorm."""


class Dropout(_UnaryIrrepsIO, _Dropout):
    """Raw-array compatible Dropout."""


class Gate(_UnaryIrrepsIO, _Gate):
    """Raw-array compatible Gate."""


class Identity(_UnaryIrrepsIO, _Identity):
    """Raw-array compatible Identity."""


class NormActivation(_UnaryIrrepsIO, _NormActivation):
    """Raw-array compatible NormActivation."""


class Extract(_Extract):
    def __call__(self, features):
        typed, raw = ensure_irreps_array(features, self.irreps_in)
        return unwrap_irreps_arrays(super().__call__(typed), raw)

    forward = __call__


class ExtractIr(_ExtractIr):
    def __call__(self, features):
        typed, raw = ensure_irreps_array(features, self.irreps_in)
        return unwrap_irreps_arrays(super().__call__(typed), raw)

    forward = __call__


class FullyConnectedNet(_FullyConnectedNet):
    forward = _FullyConnectedNet.__call__


class S2Activation(_S2Activation):
    forward = _S2Activation.__call__


class SO3Activation(_SO3Activation):
    forward = _SO3Activation.__call__


from . import models


__all__ = [
    "Activation",
    "BatchNorm",
    "Dropout",
    "Extract",
    "ExtractIr",
    "FullyConnectedNet",
    "Gate",
    "Identity",
    "NormActivation",
    "S2Activation",
    "SO3Activation",
    "models",
]
