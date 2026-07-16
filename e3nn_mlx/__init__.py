"""MLX-native e3nn runtime."""

from e3nn_core import Irrep, Irreps, MulIrrep, su2_generators, wigner_3j

from .backend import MLXBackend, mlx_backend
from .irreps_array import IrrepsArray
from .nn_gate import Gate
from .nn_identity import Identity
from .nn_activation import Activation
from .nn_dropout import Dropout
from .nn_extract import Extract, ExtractIr
from .nn_batchnorm import BatchNorm
from .nn_fc import FullyConnectedNet
from .nn_normact import NormActivation
from .nn_s2act import S2Activation
from .nn_so3act import SO3Activation
from .nn_linear import Linear
from .nn_norm import Norm
from .ops_rotations import (
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
    irreps_wigner_d,
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
from .ops_sh import (
    SphericalHarmonics,
    SphericalHarmonicsAlphaBeta,
    sh,
    spherical_harmonics,
    spherical_harmonics_alpha_beta,
)
from .ops_reduce_tensor import ReducedTensorProducts
from .ops_s2 import FromS2Grid, ToS2Grid, s2_grid
from .ops_so3 import SO3Grid, so3_irreps
from .graph import radius_graph, scatter_sum
from .radial import smooth_cutoff, soft_one_hot_linspace, soft_unit_step
from .models.gate_points_2102 import Convolution as GatePointsConvolution
from .models.gate_points_2102 import Network as GatePointsNetwork
from .models.v2106 import Convolution as V2106Convolution
from .models.v2106 import MessagePassing as V2106MessagePassing
from .models.v2106 import NetworkForAGraphWithAttributes, SimpleNetwork
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
    "Activation",
    "BatchNorm",
    "Dropout",
    "Extract",
    "ExtractIr",
    "FullTensorProduct",
    "FullyConnectedTensorProduct",
    "FullyConnectedNet",
    "FromS2Grid",
    "Gate",
    "GatePointsConvolution",
    "GatePointsNetwork",
    "Identity",
    "Irrep",
    "Irreps",
    "IrrepsArray",
    "Linear",
    "MLXBackend",
    "MulIrrep",
    "Norm",
    "NormActivation",
    "NetworkForAGraphWithAttributes",
    "S2Activation",
    "SO3Activation",
    "SO3Grid",
    "ReducedTensorProducts",
    "SphericalHarmonics",
    "SphericalHarmonicsAlphaBeta",
    "SimpleNetwork",
    "TensorProduct",
    "TensorSquare",
    "ToS2Grid",
    "V2106Convolution",
    "V2106MessagePassing",
    "angles_to_axis_angle",
    "angles_to_matrix",
    "angles_to_quaternion",
    "angles_to_xyz",
    "axis_angle_to_angles",
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "compose_angles",
    "compose_axis_angle",
    "compose_quaternion",
    "identity_angles",
    "inverse_angles",
    "irreps_wigner_d",
    "matrix_to_angles",
    "matrix_to_axis_angle",
    "matrix_to_quaternion",
    "matrix_x",
    "matrix_y",
    "matrix_z",
    "mlx_backend",
    "rotation_matrix",
    "radius_graph",
    "s2_grid",
    "scatter_sum",
    "smooth_cutoff",
    "soft_one_hot_linspace",
    "soft_unit_step",
    "so3_irreps",
    "quaternion_to_angles",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "rand_angles",
    "rand_axis_angle",
    "rand_matrix",
    "rand_quaternion",
    "sh",
    "spherical_harmonics",
    "spherical_harmonics_alpha_beta",
    "su2_generators",
    "tensor_product",
    "tensor_product_plan",
    "wigner_3j",
    "wigner_d",
    "xyz_to_angles",
]
