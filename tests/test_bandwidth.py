"""Tests for the STREAM-style bandwidth harness.

Timing tests cannot assert exact throughput without becoming machine-specific
and flaky. These assert the *structure* of the measurement instead: correct
traffic accounting, correct ordering, and physical plausibility bounds wide
enough to hold on any machine but tight enough to catch a real bug.
"""

from __future__ import annotations

import pytest

from prefill_decode_roofline.bandwidth import (
    KERNEL_TRAFFIC,
    BandwidthPoint,
    measure_kernel,
    sustained_bandwidth,
    sweep,
    theoretical_peak_gb_per_s,
)

MB = 1024 * 1024


def test_traffic_accounting_matches_stream_convention():
    """copy/scale move 2 array-lengths, add/triad move 3."""
    for kernel, arrays in KERNEL_TRAFFIC.items():
        point = measure_kernel(kernel, 1 * MB, repeats=1)
        assert point.traffic_bytes == arrays * point.array_bytes


def test_array_bytes_rounds_to_whole_float64_elements():
    point = measure_kernel("copy", 1 * MB + 3, repeats=1)
    assert point.array_bytes % 8 == 0
    assert point.array_bytes <= 1 * MB + 3


def test_unknown_kernel_is_rejected():
    with pytest.raises(ValueError, match="unknown kernel"):
        measure_kernel("fma", 1 * MB)


def test_array_smaller_than_one_element_is_rejected():
    with pytest.raises(ValueError, match="smaller than one float64"):
        measure_kernel("copy", 4)


def test_measured_bandwidth_is_positive_and_physically_plausible():
    """A few GB/s at minimum; below ~10 TB/s for any CPU DRAM path."""
    point = measure_kernel("copy", 16 * MB, repeats=3)
    assert 0.5 < point.gb_per_s < 10_000


def test_cache_resident_beats_dram_resident():
    """The whole reason the sweep exists: bandwidth falls off a cliff at LLC.

    A small array served from cache must outrun one that has to come from DRAM.
    Uses a generous margin so it holds on machines with unusual cache sizes.
    """
    small = measure_kernel("copy", 256 * 1024, repeats=7)
    large = measure_kernel("copy", 128 * MB, repeats=3)
    assert small.gb_per_s > large.gb_per_s


def test_sweep_covers_every_kernel_at_every_size():
    points = sweep([1 * MB, 2 * MB], kernels=("copy", "add"), repeats=1)
    assert len(points) == 4
    assert {p.kernel for p in points} == {"copy", "add"}


def test_sustained_bandwidth_ignores_cache_resident_points():
    """Only DRAM-resident points may set the bandwidth roof."""
    fast_small = BandwidthPoint("copy", 1 * MB, 2 * MB, seconds=0.0001)
    slow_large = BandwidthPoint("copy", 128 * MB, 256 * MB, seconds=0.01)
    got = sustained_bandwidth([fast_small, slow_large], min_array_bytes=64 * MB)
    assert got == pytest.approx(slow_large.gb_per_s)


def test_sustained_bandwidth_requires_a_dram_resident_point():
    small = BandwidthPoint("copy", 1 * MB, 2 * MB, seconds=0.001)
    with pytest.raises(ValueError, match="min_array_bytes"):
        sustained_bandwidth([small], min_array_bytes=64 * MB)


def test_working_set_accounts_for_all_live_arrays():
    point = BandwidthPoint("add", array_bytes=10, traffic_bytes=30, seconds=1.0)
    assert point.working_set_bytes == 30  # add keeps three arrays live


def test_theoretical_peak_matches_ddr5_4800_dual_channel():
    """2 channels x 4800 MT/s x 8 bytes = 76.8 GB/s."""
    assert theoretical_peak_gb_per_s(2, 4.8e9) == pytest.approx(76.8)
