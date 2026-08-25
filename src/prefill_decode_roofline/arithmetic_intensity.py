"""Arithmetic intensity of transformer prefill and decode.

Arithmetic intensity (AI) is FLOPs performed per byte moved from memory. It is
the x-axis of a roofline plot, and it is what decides whether a phase is
limited by the machine's arithmetic units or by its memory system.

The central asymmetry of autoregressive inference:

- **Prefill** processes all ``S`` prompt tokens at once. Every weight loaded
  from memory is reused across all ``S`` tokens, so AI grows with ``S``.
- **Decode** produces one token per step. Every weight is loaded to serve a
  single token, so AI is roughly constant and small no matter how large the
  model is.

Loading a weight to do one multiply-accumulate is 2 FLOPs per ~2 bytes (fp16):
an intensity near 1. No memory system in production has a ridge point that low,
so decode lands on the bandwidth roof by construction. This is why the claim
"decode is memory-bandwidth-bound" is architecture-invariant -- it follows from
batch size 1, not from any particular chip.

Counting convention: one multiply-accumulate is 2 FLOPs. We count the dense
matmuls (QKV projection, attention output, and the two MLP layers) plus the
attention score/value products. We do not count softmax, layernorm, residual
adds, or activation functions -- they are elementwise and contribute a few
percent of FLOPs. They do move bytes, so ignoring them makes the reported AI a
slight *over*-estimate; the effect is far too small to move a phase across a
ridge point that sits orders of magnitude away.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDims:
    """Dense decoder-only transformer dimensions.

    Defaults are GPT-2 124M, chosen because the numbers are checkable against a
    published model rather than invented.
    """

    name: str = "gpt2-124M"
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    vocab_size: int = 50257
    bytes_per_param: int = 2
    """fp16/bf16 serving weights."""

    @property
    def params_per_layer(self) -> int:
        """Attention (QKV + output) and MLP (up + down) projection weights."""
        attn = 4 * self.d_model * self.d_model
        mlp = 2 * self.d_model * self.d_ff
        return attn + mlp

    @property
    def transformer_params(self) -> int:
        """Weights in the transformer stack, excluding embeddings."""
        return self.n_layers * self.params_per_layer

    @property
    def weight_bytes(self) -> int:
        return self.transformer_params * self.bytes_per_param

    def kv_cache_bytes(self, seq_len: int, batch: int = 1) -> int:
        """Bytes of KV cache for ``seq_len`` tokens.

        Two tensors (K and V) per layer, each ``d_model`` wide per token.
        """
        return 2 * self.n_layers * self.d_model * seq_len * batch * self.bytes_per_param


@dataclass(frozen=True)
class PhasePoint:
    """Arithmetic intensity of one inference phase."""

    phase: str
    flops: int
    bytes_moved: int
    seq_len: int
    batch: int

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte."""
        return self.flops / self.bytes_moved


def prefill(dims: ModelDims, seq_len: int, batch: int = 1) -> PhasePoint:
    """Prefill: one forward pass over ``seq_len`` tokens at once.

    FLOPs are dominated by the dense projections, ``2 * params`` per token, plus
    attention score and value products which grow with ``seq_len``.

    Bytes are the weights (read once, reused across every token) plus the KV
    cache written out for the whole prompt.
    """
    tokens = seq_len * batch
    proj_flops = 2 * dims.transformer_params * tokens
    # QK^T and (attn @ V): 2 * 2 * seq_len^2 * d_model per layer, per sequence.
    attn_flops = dims.n_layers * batch * 4 * seq_len * seq_len * dims.d_model
    flops = proj_flops + attn_flops

    bytes_moved = dims.weight_bytes + dims.kv_cache_bytes(seq_len, batch)
    return PhasePoint("prefill", flops, bytes_moved, seq_len, batch)


def decode(dims: ModelDims, seq_len: int, batch: int = 1) -> PhasePoint:
    """Decode: generate a single token, attending over ``seq_len`` of context.

    FLOPs are ``2 * params`` per token in the batch -- one token each. Bytes are
    the full weight set (re-read for this one step) plus the KV cache read back
    to attend over the context.

    The weights term dominates unless the context is very long, which is the
    whole point: batch 1 decode pays the entire model's memory traffic to
    produce one token.
    """
    proj_flops = 2 * dims.transformer_params * batch
    attn_flops = dims.n_layers * batch * 4 * seq_len * dims.d_model
    flops = proj_flops + attn_flops

    bytes_moved = dims.weight_bytes + dims.kv_cache_bytes(seq_len, batch)
    return PhasePoint("decode", flops, bytes_moved, seq_len, batch)


def ridge_point(peak_flops_per_s: float, peak_bytes_per_s: float) -> float:
    """Arithmetic intensity where the compute and bandwidth roofs meet.

    Below it a kernel is memory-bound, above it compute-bound. It is a property
    of the machine alone -- no workload appears in the formula.
    """
    return peak_flops_per_s / peak_bytes_per_s


def is_memory_bound(point: PhasePoint, ridge: float) -> bool:
    return point.arithmetic_intensity < ridge


def attainable_flops_per_s(
    intensity: float, peak_flops_per_s: float, peak_bytes_per_s: float
) -> float:
    """The roofline itself: ``min(peak_compute, bandwidth * intensity)``."""
    return min(peak_flops_per_s, peak_bytes_per_s * intensity)
