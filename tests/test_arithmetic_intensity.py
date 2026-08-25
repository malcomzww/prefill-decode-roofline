"""Tests for the analytical arithmetic-intensity model.

These are the tests that matter most: the AI model is what carries the repo's
central claim, and unlike the timing code it is exactly checkable.
"""

from __future__ import annotations

import pytest

from prefill_decode_roofline.arithmetic_intensity import (
    ModelDims,
    attainable_flops_per_s,
    decode,
    is_memory_bound,
    prefill,
    ridge_point,
)


def test_gpt2_dimensions_reconstruct_published_parameter_count():
    """GPT-2 124M = 84.9M transformer weights + 39.4M embeddings.

    Anchors the dimensions to a real published model so a typo in d_model or
    n_layers fails here rather than silently skewing every downstream number.
    """
    d = ModelDims()
    embeddings = (d.vocab_size + 1024) * d.d_model
    total = d.transformer_params + embeddings
    assert d.transformer_params == pytest.approx(84.9e6, rel=0.01)
    assert total == pytest.approx(124.3e6, rel=0.01)


def test_decode_intensity_is_about_one_flop_per_byte():
    """Two FLOPs per multiply-accumulate over a 2-byte weight -> AI ~ 1.

    This is the structural heart of the repo. Batch-1 decode reads each weight
    to do exactly one MAC with it.
    """
    d = ModelDims()
    assert decode(d, seq_len=512).arithmetic_intensity == pytest.approx(1.0, rel=0.02)


def test_decode_intensity_is_invariant_to_model_size():
    """A 7B model has the same decode intensity as a 124M one.

    The claim "decode is memory-bandwidth-bound" does not depend on scale --
    which is why it transfers across architectures.
    """
    small = decode(ModelDims(), seq_len=1024).arithmetic_intensity
    big = decode(
        ModelDims(name="7b", n_layers=32, d_model=4096, n_heads=32, d_ff=11008),
        seq_len=1024,
    ).arithmetic_intensity
    assert small == pytest.approx(big, rel=0.05)


def test_prefill_intensity_grows_with_sequence_length():
    """Weights are amortised over S tokens, so AI rises roughly linearly."""
    d = ModelDims()
    ais = [prefill(d, seq_len=s).arithmetic_intensity for s in (128, 256, 512, 1024)]
    assert ais == sorted(ais)
    assert ais[-1] > 4 * ais[0]


def test_prefill_exceeds_decode_intensity_by_orders_of_magnitude():
    d = ModelDims()
    assert prefill(d, 512).arithmetic_intensity > 100 * decode(d, 512).arithmetic_intensity


def test_batching_raises_decode_intensity_sublinearly():
    """Continuous batching amortises one weight read over B tokens.

    The gain is strictly sublinear in B, and this test pins down why: the KV
    cache scales with batch too, so batching amortises the *weights* but not
    the cache. At S=1024 a 32x batch buys only ~4.8x intensity. Short context
    is closer to linear because the weight term dominates traffic there.
    """
    d = ModelDims()
    b1 = decode(d, 1024, batch=1).arithmetic_intensity
    b32 = decode(d, 1024, batch=32).arithmetic_intensity
    assert 1.0 < b32 / b1 < 32.0
    assert b32 / b1 == pytest.approx(4.8, rel=0.05)

    # With a short context the KV term is negligible, so the same 32x batch
    # gets much closer to the ideal linear speedup.
    short_gain = decode(d, 64, batch=32).arithmetic_intensity / decode(
        d, 64, batch=1
    ).arithmetic_intensity
    assert short_gain > b32 / b1


def test_batching_cannot_lift_decode_to_a_gpu_ridge_point():
    """Even 32x batching leaves decode memory-bound on an accelerator.

    Guards against over-claiming that batching "fixes" decode: it moves the
    point right along the bandwidth roof, it does not cross the ridge.
    """
    d = ModelDims()
    assert is_memory_bound(decode(d, 1024, batch=32), ridge=300.0)


def test_decode_is_memory_bound_across_plausible_ridge_points():
    """Decode sits below the ridge on DDR5 and on HBM alike.

    Ridge points spanning CPU (~25) through modern accelerators (several
    hundred) all leave batch-1 decode memory-bound.
    """
    d = ModelDims()
    point = decode(d, 1024)
    for ridge in (10.0, 25.0, 100.0, 300.0, 600.0):
        assert is_memory_bound(point, ridge)


def test_ridge_point_is_ratio_of_the_two_roofs():
    assert ridge_point(800e9, 32e9) == pytest.approx(25.0)


def test_attainable_is_bandwidth_limited_below_ridge_and_flat_above():
    peak_f, peak_b = 800e9, 32e9
    ridge = ridge_point(peak_f, peak_b)
    below = attainable_flops_per_s(ridge / 10, peak_f, peak_b)
    above = attainable_flops_per_s(ridge * 10, peak_f, peak_b)
    assert below == pytest.approx(peak_b * ridge / 10)
    assert above == pytest.approx(peak_f)


def test_kv_cache_grows_linearly_with_context_and_batch():
    d = ModelDims()
    assert d.kv_cache_bytes(2048) == 2 * d.kv_cache_bytes(1024)
    assert d.kv_cache_bytes(512, batch=4) == 4 * d.kv_cache_bytes(512)


def test_long_context_shifts_decode_traffic_toward_the_kv_cache():
    """At short context weights dominate traffic; at long context KV catches up."""
    d = ModelDims()
    short = decode(d, 128)
    long = decode(d, 16384)
    assert d.kv_cache_bytes(128) < 0.1 * d.weight_bytes
    assert d.kv_cache_bytes(16384) > d.weight_bytes
    assert long.bytes_moved > short.bytes_moved
