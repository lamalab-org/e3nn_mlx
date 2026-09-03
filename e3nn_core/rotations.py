"""Static rotation metadata."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class RotationAngles:
    alpha: float
    beta: float
    gamma: float
    convention: str = "yxy"

    def normalized(self) -> RotationAngles:
        return RotationAngles(
            alpha=math.fmod(self.alpha, 2.0 * math.pi),
            beta=math.fmod(self.beta, 2.0 * math.pi),
            gamma=math.fmod(self.gamma, 2.0 * math.pi),
            convention=self.convention,
        )
