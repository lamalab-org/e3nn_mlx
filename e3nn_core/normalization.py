"""Normalization metadata for symbolic operators."""

from __future__ import annotations

from dataclasses import dataclass

from .typing import IrrepNormalizationMode, PathNormalizationMode


@dataclass(frozen=True, slots=True)
class NormalizationMetadata:
    irrep_normalization: IrrepNormalizationMode = "component"
    path_normalization: PathNormalizationMode = "element"
    num_paths: int = 1
    num_elements: int = 1
    coefficient: float = 1.0

    def scale(self) -> float:
        return self.coefficient
