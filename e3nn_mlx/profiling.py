"""Profiling helpers."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from .ops_basic import compile_or_identity, eval_maybe


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    eager_ms: float
    compiled_ms: float

    @property
    def speedup(self) -> float:
        if self.compiled_ms == 0.0:
            return float("inf")
        return self.eager_ms / self.compiled_ms


def benchmark_callable(fn: Callable[..., Any], *args: Any, warmup: int = 1, iters: int = 5, **kwargs: Any) -> float:
    for _ in range(warmup):
        eval_maybe(fn(*args, **kwargs))
    start = time.perf_counter()
    for _ in range(iters):
        eval_maybe(fn(*args, **kwargs))
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iters


def compare_eager_and_compiled(name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> BenchmarkResult:
    eager_ms = benchmark_callable(fn, *args, **kwargs)
    compiled = compile_or_identity(fn, enabled=True)
    compiled_ms = benchmark_callable(compiled, *args, **kwargs)
    return BenchmarkResult(name=name, eager_ms=eager_ms, compiled_ms=compiled_ms)
