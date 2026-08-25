"""Tests for the GEMM compute-roof harness."""

from __future__ import annotations

import pytest

from prefill_decode_roofline.flops import GemmPoint, measure_gemm, peak_gflops, sweep


def test_gemm_flop_count_is_two_n_cubed():
    assert GemmPoint(n=1024, dtype="float32", seconds=1.0).flops == 2 * 1024**3


def test_gemm_arithmetic_intensity_grows_with_n():
    """AI ~ 2N^3 / 3N^2 = O(N): big GEMMs are compute-bound, which is why
    they are the right probe for the compute roof."""
    ais = [GemmPoint(n=n, dtype="float32", seconds=1.0).arithmetic_intensity for n in (256, 1024)]
    assert ais[1] == pytest.approx(4 * ais[0])


def test_gemm_arithmetic_intensity_is_well_above_a_cpu_ridge_point():
    point = GemmPoint(n=2048, dtype="float32", seconds=1.0)
    assert point.arithmetic_intensity > 100


def test_fp64_has_half_the_intensity_of_fp32():
    """Same FLOPs, twice the bytes."""
    f32 = GemmPoint(n=512, dtype="float32", seconds=1.0).arithmetic_intensity
    f64 = GemmPoint(n=512, dtype="float64", seconds=1.0).arithmetic_intensity
    assert f64 == pytest.approx(f32 / 2)


def test_measured_gemm_is_plausible_and_deterministic_in_shape():
    point = measure_gemm(256, repeats=2, seed=0)
    assert point.n == 256
    assert point.seconds > 0
    assert 0.1 < point.gflops_per_s < 1e6


def test_larger_gemm_reaches_higher_throughput_than_a_tiny_one():
    """Small matrices cannot saturate the pipeline; this is why peak_gflops
    takes a max over sizes rather than an average."""
    small = measure_gemm(64, repeats=3, seed=0)
    large = measure_gemm(1024, repeats=3, seed=0)
    assert large.gflops_per_s > small.gflops_per_s


def test_peak_takes_the_maximum_across_the_sweep():
    points = [
        GemmPoint(n=512, dtype="float32", seconds=1.0),
        GemmPoint(n=512, dtype="float32", seconds=0.5),
    ]
    assert peak_gflops(points) == pytest.approx(points[1].gflops_per_s)


def test_peak_requires_at_least_one_point():
    with pytest.raises(ValueError, match="no GEMM points"):
        peak_gflops([])


def test_sweep_returns_one_point_per_size():
    points = sweep([128, 256], repeats=1)
    assert [p.n for p in points] == [128, 256]
