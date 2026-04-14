"""Shared type aliases for backend-neutral metadata."""

from __future__ import annotations

from typing import Literal, TypeAlias

Parity: TypeAlias = Literal[-1, 1]
NormalizationMode: TypeAlias = Literal["component", "norm", "integral"]
IrrepNormalizationMode: TypeAlias = Literal["component", "norm", "none"]
PathNormalizationMode: TypeAlias = Literal["element", "path", "none", "component"]
TensorProductMode: TypeAlias = Literal["uuu", "uvu", "uvv", "uvw", "uuw", "uvuv", "uvu<v", "u<vw"]
