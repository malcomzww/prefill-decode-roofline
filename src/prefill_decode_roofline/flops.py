"""Measured peak floating-point throughput via dense GEMM.

Matrix multiply is the right probe for the compute roof. An N x N GEMM does
``2N^3`` FLOPs against ``3N^2`` elements of memory traffic, so arithmetic
intensity grows as O(N): at large N the kernel is firmly compute-bound and what
you measure is the arithmetic units, not the memory system.

We use numpy, which dispatches to the platform BLAS. torch's CPU GEMM was also
measured during development and was both slower and far less stable run to run
(it fell to a quarter of numpy's throughput at N=4096 on this machine, under
memory pressure and thread contention). The roofline needs the machine's *best*
sustained rate, so the faster and more reproducible of the two is the honest
choice for a ceiling.

We report the maximum across sizes rather than a mean. A roofline ceiling is a
capability claim -- "the machine can reach this" -- and averaging in the small
sizes that have not yet saturated the pipeline would understate it.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GemmPoint:
    """One square-GEMM measurement."""

    n: int
    dtype: str
    seconds: float
    """Fastest observed time for one multiply."""

    @property
    def flops(self) -> int:
        """2 N^3: one multiply and one add per inner-product term."""
        return 2 * self.n**3

    @property
    def gflops_per_s(self) -> float:
        return self.flops / self.seconds / 1e9

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte, counting the three matrices touched once each."""
        itemsize = np.dtype(self.dtype).itemsize
        return self.flops / (3 * self.n**2 * itemsize)


def measure_gemm(n: int, dtype: str = "float32", repeats: int = 5, seed: int = 0) -> GemmPoint:
    """Time an ``n x n`` matrix multiply, reporting the fastest run."""
    rng = np.random.default_rng(seed)
    a = rng.random((n, n), dtype=np.float32).astype(dtype)
    b = rng.random((n, n), dtype=np.float32).astype(dtype)

    a @ b  # warm up BLAS thread pool and first-touch the pages
    a @ b

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter_ns()
        a @ b
        best = min(best, (time.perf_counter_ns() - start) / 1e9)

    del a, b
    return GemmPoint(n=n, dtype=dtype, seconds=best)


def sweep(sizes: list[int], dtype: str = "float32", repeats: int = 5) -> list[GemmPoint]:
    """Measure each size, releasing memory between sizes.

    The explicit ``gc.collect()`` is load-bearing on a memory-constrained
    machine. Without it, a sweep leaves each size's matrices reachable long
    enough that later sizes allocate under pressure and degrade into paging:
    during development N=4096 measured ~870 GFLOP/s in isolation but ~320 at
    the end of a sweep, a 2.7x error that would have silently understated the
    compute roof and dragged the ridge point down with it.
    """
    points = []
    for n in sizes:
        points.append(measure_gemm(n, dtype=dtype, repeats=repeats))
        gc.collect()
    return points


def peak_gflops(points: list[GemmPoint]) -> float:
    """Best sustained rate observed -- the compute roof."""
    if not points:
        raise ValueError("no GEMM points")
    return max(p.gflops_per_s for p in points)
