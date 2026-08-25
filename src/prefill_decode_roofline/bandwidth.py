"""STREAM-style memory bandwidth measurement.

The classic STREAM benchmark (McCalpin, 1995) measures sustainable memory
bandwidth with four kernels that do almost no arithmetic, so the memory system
is the only thing under test:

    copy   c[i] = a[i]              2 array-lengths of traffic
    scale  b[i] = s * c[i]          2
    add    c[i] = a[i] + b[i]       3
    triad  a[i] = b[i] + s * c[i]   3

numpy has no fused triad: `a = b + s*c` is two ufunc calls, so the scratch
buffer holding `s*c` is written then re-read. That extra round trip is real
traffic we do not count, which is why triad reads lower than add here rather
than matching it as it does in compiled STREAM. It is a numpy artifact, not a
property of the memory system.

Counting traffic is the subtle part. We count *compulsory* traffic -- the bytes
the kernel must logically read and write -- which is what STREAM reports. On a
write-allocate cache the hardware also reads the destination line before
overwriting it, so the true DRAM traffic for ``copy`` is closer to 3 array
lengths than 2. That means these figures, like STREAM's, are a conservative
*lower bound* on the bandwidth the memory controller actually delivered. We do
not correct for it, because whether a given store is write-allocating or
non-temporal depends on what numpy's compiler emitted, and guessing would be
inventing a number.

Timing uses the *minimum* over repetitions rather than the mean. The minimum is
the run least disturbed by scheduler preemption and other processes; on a
shared laptop the mean measures the rest of the system as much as the kernel.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

BYTES_PER_F64 = 8


@dataclass(frozen=True)
class BandwidthPoint:
    """One (kernel, working-set size) measurement."""

    kernel: str
    array_bytes: int
    """Size of a single array, the number that decides cache residency."""
    traffic_bytes: int
    """Compulsory bytes moved by one kernel invocation."""
    seconds: float
    """Fastest observed time for one invocation."""

    @property
    def gb_per_s(self) -> float:
        return self.traffic_bytes / self.seconds / 1e9

    @property
    def working_set_bytes(self) -> int:
        """Total footprint of all arrays the kernel touches."""
        return self.array_bytes * KERNEL_ARRAYS[self.kernel]


# Arrays each kernel keeps live, and array-lengths of compulsory traffic.
KERNEL_ARRAYS = {"copy": 2, "scale": 2, "add": 3, "triad": 3}
KERNEL_TRAFFIC = {"copy": 2, "scale": 2, "add": 3, "triad": 3}


def _time_min(fn: Callable[[], object], repeats: int, warmup: int = 2) -> float:
    """Fastest of ``repeats`` runs, in seconds, after warming caches."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        best = min(best, (time.perf_counter_ns() - start) / 1e9)
    return best


def measure_kernel(kernel: str, array_bytes: int, repeats: int = 7) -> BandwidthPoint:
    """Measure one STREAM kernel at one working-set size.

    ``array_bytes`` is rounded down to a whole number of float64 elements.
    """
    if kernel not in KERNEL_ARRAYS:
        raise ValueError(f"unknown kernel {kernel!r}; expected one of {sorted(KERNEL_ARRAYS)}")
    n = array_bytes // BYTES_PER_F64
    if n < 1:
        raise ValueError(f"array_bytes={array_bytes} is smaller than one float64 element")

    scalar = 3.0
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)
    c = np.ones(n, dtype=np.float64)

    if kernel == "copy":
        fn: Callable[[], object] = lambda: np.copyto(c, a)  # noqa: E731
    elif kernel == "scale":
        fn = lambda: np.multiply(c, scalar, out=b)  # noqa: E731
    elif kernel == "add":
        fn = lambda: np.add(a, b, out=c)  # noqa: E731
    else:
        # triad: a = b + s * c. Both ufuncs write through `out=` into
        # preallocated buffers. Writing `np.add(b, np.multiply(c, scalar))`
        # instead allocates a fresh array of the full working-set size on every
        # call, which measures the allocator rather than the memory system --
        # it read ~4x slower here before this was fixed.
        scratch = np.empty_like(c)

        def fn() -> object:
            np.multiply(c, scalar, out=scratch)
            return np.add(b, scratch, out=a)

    seconds = _time_min(fn, repeats=repeats)
    return BandwidthPoint(
        kernel=kernel,
        array_bytes=n * BYTES_PER_F64,
        traffic_bytes=KERNEL_TRAFFIC[kernel] * n * BYTES_PER_F64,
        seconds=seconds,
    )


def sweep(
    array_bytes_list: list[int],
    kernels: tuple[str, ...] = ("copy", "scale", "add", "triad"),
    repeats: int = 7,
) -> list[BandwidthPoint]:
    """Measure every kernel at every size."""
    return [
        measure_kernel(k, size, repeats=repeats) for size in array_bytes_list for k in kernels
    ]


def sustained_bandwidth(points: list[BandwidthPoint], min_array_bytes: int) -> float:
    """Peak GB/s among points whose arrays are too big to sit in cache.

    This is the DRAM-resident plateau: the bandwidth that survives when the
    working set no longer fits in last-level cache. Callers pass a
    ``min_array_bytes`` comfortably above LLC size.
    """
    dram = [p for p in points if p.array_bytes >= min_array_bytes]
    if not dram:
        raise ValueError("no points at or above min_array_bytes")
    return max(p.gb_per_s for p in dram)


def theoretical_peak_gb_per_s(
    channels: int, transfers_per_second: float, bus_width_bits: int = 64
) -> float:
    """Datasheet peak: channels x MT/s x bus width.

    For DDR5-4800 on two 64-bit channels: 2 x 4.8e9 x 8 = 76.8 GB/s. This is an
    upper bound no benchmark can reach -- it assumes zero refresh, zero
    page-miss penalty, and perfect read/write turnaround.
    """
    return channels * transfers_per_second * (bus_width_bits / 8) / 1e9
