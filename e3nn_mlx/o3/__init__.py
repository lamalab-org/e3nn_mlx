"""e3nn-compatible :mod:`o3` namespace backed by MLX.

Classes in this namespace accept raw MLX arrays like upstream PyTorch/e3nn.
Passing :class:`IrrepsArray` keeps the representation-aware return type.
"""

from __future__ import annotations

from typing import Any

from e3nn_core import (
    Irrep,
    Irreps,
    MulIrrep,
    TensorProductInstruction as Instruction,
    change_basis_real_to_complex,
    so3_generators,
    su2_generators,
    wigner_3j,
)

from .._high_level import ensure_irreps_array, unwrap_irreps_arrays
from ..irreps_array import IrrepsArray
from ..nn_linear import Linear as _Linear
from ..nn_norm import Norm as _Norm
from ..ops_reduce_tensor import ReducedTensorProducts as _ReducedTensorProducts
from ..ops_rotations import (
    angles_to_axis_angle,
    angles_to_matrix,
    angles_to_quaternion,
    angles_to_xyz,
    axis_angle_to_angles,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    compose_angles,
    compose_axis_angle,
    compose_quaternion,
    identity_angles,
    inverse_angles,
    irrep_wigner_d,
    irrep_wigner_d_from_axis_angle,
    irrep_wigner_d_from_matrix,
    irrep_wigner_d_from_quaternion,
    irreps_randn,
    irreps_wigner_d,
    irreps_wigner_d_from_axis_angle,
    irreps_wigner_d_from_matrix,
    irreps_wigner_d_from_quaternion,
    matrix_to_angles,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_x,
    matrix_y,
    matrix_z,
    quaternion_to_angles,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rand_angles,
    rand_axis_angle,
    rand_matrix,
    rand_quaternion,
    rotation_matrix,
    wigner_d,
    xyz_to_angles,
)
from ..ops_s2 import FromS2Grid, ToS2Grid, s2_grid
from ..ops_sh import (
    SphericalHarmonics as _SphericalHarmonics,
    SphericalHarmonicsAlphaBeta as _SphericalHarmonicsAlphaBeta,
    spherical_harmonics,
    spherical_harmonics_alpha_beta,
)
from ..ops_so3 import SO3Grid
from ..ops_tp import (
    ElementwiseTensorProduct as _ElementwiseTensorProduct,
    FullTensorProduct as _FullTensorProduct,
    FullyConnectedTensorProduct as _FullyConnectedTensorProduct,
    TensorProduct as _TensorProduct,
    TensorSquare as _TensorSquare,
)


wigner_D = wigner_d


class Linear(_Linear):
    def __call__(
        self,
        features,
        weight: Any | None = None,
        *,
        bias: Any | None = None,
        extension: str | None = None,
    ):
        typed, raw = ensure_irreps_array(features, self.irreps_in)
        output = super().__call__(
            typed,
            weight,
            bias=bias,
            extension=extension,
        )
        return unwrap_irreps_arrays(output, raw)

    forward = __call__


class Norm(_Norm):
    def __call__(self, features):
        if self.irreps_in is None and not isinstance(features, IrrepsArray):
            raise TypeError("raw MLX inputs require Norm(irreps_in=...)")
        irreps = features.irreps if isinstance(features, IrrepsArray) else self.irreps_in
        typed, raw = ensure_irreps_array(features, irreps)
        return unwrap_irreps_arrays(super().__call__(typed), raw)

    forward = __call__


class _TensorProductIO:
    def __call__(self, left, right, weight: Any | None = None):
        typed_left, raw_left = ensure_irreps_array(left, self.irreps_in1)
        typed_right, raw_right = ensure_irreps_array(right, self.irreps_in2)
        output = super().__call__(typed_left, typed_right, weight)
        return unwrap_irreps_arrays(output, raw_left and raw_right)

    def forward(self, left, right, weight: Any | None = None):
        return self(left, right, weight)

    def right(self, right, weight: Any | None = None):
        typed, _ = ensure_irreps_array(right, self.irreps_in2)
        return super().right(typed, weight)


class TensorProduct(_TensorProductIO, _TensorProduct):
    """Raw-array compatible TensorProduct."""


class FullyConnectedTensorProduct(_TensorProductIO, _FullyConnectedTensorProduct):
    """Raw-array compatible FullyConnectedTensorProduct."""


class ElementwiseTensorProduct(_TensorProductIO, _ElementwiseTensorProduct):
    """Raw-array compatible ElementwiseTensorProduct."""


class FullTensorProduct(_TensorProductIO, _FullTensorProduct):
    """Raw-array compatible FullTensorProduct."""


class TensorSquare(_TensorSquare):
    def __call__(self, features, weight: Any | None = None):
        typed, raw = ensure_irreps_array(features, self.irreps_in)
        return unwrap_irreps_arrays(super().__call__(typed, weight), raw)

    forward = __call__


class ReducedTensorProducts(_ReducedTensorProducts):
    def __call__(self, *inputs):
        if len(inputs) != len(self.irreps_in):
            return super().__call__(*inputs)
        converted = [
            ensure_irreps_array(value, irreps)
            for value, irreps in zip(inputs, self.irreps_in, strict=True)
        ]
        output = super().__call__(*(value for value, _ in converted))
        return unwrap_irreps_arrays(output, all(raw for _, raw in converted))

    forward = __call__


class SphericalHarmonics(_SphericalHarmonics):
    forward = _SphericalHarmonics.__call__


class SphericalHarmonicsAlphaBeta(_SphericalHarmonicsAlphaBeta):
    forward = _SphericalHarmonicsAlphaBeta.__call__


__all__ = [
    "ElementwiseTensorProduct",
    "FromS2Grid",
    "FullTensorProduct",
    "FullyConnectedTensorProduct",
    "Instruction",
    "Irrep",
    "Irreps",
    "IrrepsArray",
    "Linear",
    "MulIrrep",
    "Norm",
    "ReducedTensorProducts",
    "SphericalHarmonics",
    "SphericalHarmonicsAlphaBeta",
    "SO3Grid",
    "TensorProduct",
    "TensorSquare",
    "ToS2Grid",
    "angles_to_axis_angle",
    "angles_to_matrix",
    "angles_to_quaternion",
    "angles_to_xyz",
    "axis_angle_to_angles",
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "change_basis_real_to_complex",
    "compose_angles",
    "compose_axis_angle",
    "compose_quaternion",
    "identity_angles",
    "inverse_angles",
    "irrep_wigner_d",
    "irrep_wigner_d_from_axis_angle",
    "irrep_wigner_d_from_matrix",
    "irrep_wigner_d_from_quaternion",
    "irreps_randn",
    "irreps_wigner_d",
    "irreps_wigner_d_from_axis_angle",
    "irreps_wigner_d_from_matrix",
    "irreps_wigner_d_from_quaternion",
    "matrix_to_angles",
    "matrix_to_axis_angle",
    "matrix_to_quaternion",
    "matrix_x",
    "matrix_y",
    "matrix_z",
    "quaternion_to_angles",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "rand_angles",
    "rand_axis_angle",
    "rand_matrix",
    "rand_quaternion",
    "rotation_matrix",
    "s2_grid",
    "so3_generators",
    "spherical_harmonics",
    "spherical_harmonics_alpha_beta",
    "su2_generators",
    "wigner_3j",
    "wigner_D",
    "wigner_d",
    "xyz_to_angles",
]
