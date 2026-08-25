"""Tests for the assembled roofline model."""

from __future__ import annotations

import pytest

from prefill_decode_roofline.arithmetic_intensity import ModelDims, decode, prefill
from prefill_decode_roofline.roofline import Roofline, plot

# Roughly this machine: ~860 GFLOP/s compute roof, ~34 GB/s DRAM roof.
MACHINE = Roofline(peak_flops_per_s=860e9, peak_bytes_per_s=34e9)


def test_ridge_point_is_where_the_two_roofs_cross():
    assert MACHINE.ridge == pytest.approx(860e9 / 34e9)
    at_ridge = MACHINE.attainable(MACHINE.ridge)
    assert at_ridge == pytest.approx(MACHINE.peak_flops_per_s, rel=1e-9)


def test_attainable_never_exceeds_either_roof():
    for intensity in (0.1, 1, 10, 25, 100, 1e4):
        got = MACHINE.attainable(intensity)
        assert got <= MACHINE.peak_flops_per_s * (1 + 1e-9)
        assert got <= MACHINE.peak_bytes_per_s * intensity * (1 + 1e-9)


def test_bound_by_flips_at_the_ridge():
    assert MACHINE.bound_by(MACHINE.ridge / 2) == "memory"
    assert MACHINE.bound_by(MACHINE.ridge * 2) == "compute"


def test_decode_is_memory_bound_and_prefill_is_compute_bound():
    """The repo's headline claim, evaluated against a measured roofline."""
    dims = ModelDims()
    assert MACHINE.bound_by(decode(dims, 1024).arithmetic_intensity) == "memory"
    assert MACHINE.bound_by(prefill(dims, 1024).arithmetic_intensity) == "compute"


def test_decode_utilisation_is_a_few_percent_of_the_compute_roof():
    """Decode cannot use the arithmetic units, however good the kernel is.

    This is the number that makes tokens/sec a misleading headline.
    """
    util = MACHINE.utilisation(decode(ModelDims(), 1024))
    assert 0 < util < 0.10


def test_prefill_saturates_the_compute_roof():
    assert MACHINE.utilisation(prefill(ModelDims(), 1024)) == pytest.approx(1.0)


def test_the_claim_survives_a_gpu_class_roofline():
    """Same verdict on an HBM machine: only the absolute numbers move.

    ~1000 TFLOP/s against ~3.35 TB/s, an H100-class ratio.
    """
    gpu = Roofline(peak_flops_per_s=1000e12, peak_bytes_per_s=3.35e12, label="hbm-class")
    dims = ModelDims()
    assert gpu.ridge > MACHINE.ridge
    assert gpu.bound_by(decode(dims, 1024).arithmetic_intensity) == "memory"
    assert gpu.utilisation(decode(dims, 1024)) < 0.01


def test_plot_writes_a_png(tmp_path):
    dims = ModelDims()
    out = tmp_path / "roofline.png"
    returned = plot(
        MACHINE,
        [prefill(dims, 1024), decode(dims, 1024)],
        out_path=str(out),
        theoretical_bytes_per_s=76.8e9,
    )
    assert returned == str(out)
    assert out.exists()
    assert out.stat().st_size > 5_000  # a real figure, not an empty canvas
