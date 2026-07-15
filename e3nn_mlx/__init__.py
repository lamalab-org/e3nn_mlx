"""MLX-native e3nn runtime."""

from e3nn_core import Irrep, Irreps, MulIrrep, wigner_3j

from .backend import MLXBackend, mlx_backend
from .irreps_array import IrrepsArray
from .nn_gate import Gate
from .nn_linear import Linear
from .nn_norm import Norm
from .ops_rotations import irreps_wigner_d, rotation_matrix, wigner_d
from .ops_sh import sh, spherical_harmonics
from .ops_tp import (
    ElementwiseTensorProduct,
    FullTensorProduct,
    FullyConnectedTensorProduct,
    TensorProduct,
    TensorSquare,
    tensor_product,
    tensor_product_plan,
)

__all__ = [
    "ElementwiseTensorProduct",
    "FullTensorProduct",
    "FullyConnectedTensorProduct",
    "Gate",
    "Irrep",
    "Irreps",
    "IrrepsArray",
    "Linear",
    "MLXBackend",
    "MulIrrep",
    "Norm",
    "TensorProduct",
    "TensorSquare",
    "irreps_wigner_d",
    "mlx_backend",
    "rotation_matrix",
    "sh",
    "spherical_harmonics",
    "tensor_product",
    "tensor_product_plan",
    "wigner_3j",
    "wigner_d",
]
