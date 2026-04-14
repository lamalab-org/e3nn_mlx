"""Normalization metadata for symbolic operators."""

from __future__ import annotations

from dataclasses import dataclass

from .typing import NormalizationMode


@dataclass(frozen=True, slots=True)
class NormalizationMetadata:
    irrep_normalization: NormalizationMode = "component"
    path_normalization: NormalizationMode = "component"
    num_paths: int = 1

    def scale(self) -> float:
        if self.num_paths <= 0:
            raise ValueError("num_paths must be > 0")
        if self.path_normalization == "component":
            return 1.0
        return self.num_paths ** -0.5
