"""e3nn-style model namespace with raw-array compatibility adapters."""

from .gate_points_2102 import Compose, Convolution, Network, tp_path_exists
from .v2106 import (
    Convolution as V2106Convolution,
    MessagePassing as V2106MessagePassing,
    NetworkForAGraphWithAttributes,
    SimpleNetwork,
)

__all__ = [
    "Compose",
    "Convolution",
    "Network",
    "NetworkForAGraphWithAttributes",
    "SimpleNetwork",
    "V2106Convolution",
    "V2106MessagePassing",
    "tp_path_exists",
]
