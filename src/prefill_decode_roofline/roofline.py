"""Assemble measured roofs into a roofline model, and plot it.

The roofline (Williams, Waterman & Patterson, 2009) bounds attainable
performance by two measured machine constants:

    attainable_FLOP/s = min(peak_compute, peak_bandwidth x arithmetic_intensity)

The bend between the sloped bandwidth roof and the flat compute roof is the
**ridge point**. Its position is a property of the machine, and where a
workload's arithmetic intensity falls relative to it decides which resource is
the binding constraint -- and therefore which optimisation can possibly help.
"""

from __future__ import annotations

from dataclasses import dataclass

from .arithmetic_intensity import PhasePoint, attainable_flops_per_s, ridge_point


@dataclass(frozen=True)
class Roofline:
    """A machine's measured compute and bandwidth roofs."""

    peak_flops_per_s: float
    peak_bytes_per_s: float
    label: str = "measured"

    @property
    def ridge(self) -> float:
        return ridge_point(self.peak_flops_per_s, self.peak_bytes_per_s)

    def attainable(self, intensity: float) -> float:
        return attainable_flops_per_s(intensity, self.peak_flops_per_s, self.peak_bytes_per_s)

    def bound_by(self, intensity: float) -> str:
        return "memory" if intensity < self.ridge else "compute"

    def utilisation(self, point: PhasePoint) -> float:
        """Fraction of the compute roof this phase could reach at best.

        A memory-bound phase cannot reach the compute roof no matter how good
        the kernel is; this quantifies how far the roofline alone holds it back.
        """
        return self.attainable(point.arithmetic_intensity) / self.peak_flops_per_s


def plot(
    roof: Roofline,
    phases: list[PhasePoint],
    out_path: str,
    theoretical_bytes_per_s: float | None = None,
    title: str = "Roofline: prefill vs decode",
) -> str:
    """Render the roofline with each phase placed on it. Returns ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    intensities = np.logspace(-1, 4, 400)
    attainable = [roof.attainable(x) / 1e9 for x in intensities]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(intensities, attainable, linewidth=2.4, color="#1f3b73", label="measured roofline")

    if theoretical_bytes_per_s is not None:
        theo = [
            min(roof.peak_flops_per_s, theoretical_bytes_per_s * x) / 1e9 for x in intensities
        ]
        ax.loglog(
            intensities,
            theo,
            linewidth=1.5,
            linestyle="--",
            color="#9aa5b8",
            label="datasheet-peak bandwidth roof",
        )

    ax.axvline(roof.ridge, color="#b0451c", linestyle=":", linewidth=1.6)
    ax.annotate(
        f"ridge point\n{roof.ridge:.0f} FLOP/byte",
        xy=(roof.ridge, roof.peak_flops_per_s / 1e9),
        xytext=(roof.ridge * 1.35, roof.peak_flops_per_s / 1e9 * 0.16),
        color="#b0451c",
        fontsize=9,
    )

    markers = {"prefill": ("o", "#1a7f4b"), "decode": ("D", "#b0451c")}
    for p in phases:
        marker, colour = markers.get(p.phase, ("s", "#444444"))
        x = p.arithmetic_intensity
        y = roof.attainable(x) / 1e9
        ax.plot(x, y, marker, color=colour, markersize=9, zorder=5)
        ax.annotate(
            f"{p.phase} S={p.seq_len}",
            xy=(x, y),
            xytext=(0, -16 if p.phase == "decode" else 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=colour,
        )

    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("attainable performance (GFLOP/s)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=9)

    left = ax.get_xlim()[0]
    ax.axvspan(left, roof.ridge, color="#b0451c", alpha=0.05)
    ax.text(
        left * 1.5,
        roof.peak_flops_per_s / 1e9 * 0.02,
        "memory-bandwidth-bound",
        fontsize=8,
        color="#b0451c",
    )
    ax.text(
        roof.ridge * 2.2,
        roof.peak_flops_per_s / 1e9 * 0.02,
        "compute-bound",
        fontsize=8,
        color="#1a7f4b",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
