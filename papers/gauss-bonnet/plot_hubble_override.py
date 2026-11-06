#!/usr/bin/env python3
"""Plot the initial-H diagnostic for the N=400 anomaly.

`n128_h400` is the standard N=128 lattice run (same k0, dt, tolerances and
seed), except that its initial Hubble rate is forced to the value of the
existing N=400 run.  If the N=400 spectrum shift is caused by the
unrenormalized UV vacuum energy raising H_ini, then n128_h400 should move
towards n400; if it stays on top of n128, the initial H alone is not the
relevant channel.

The script is intentionally independent of `plot_convergence.py` so that the
standard convergence figures are not modified.

Usage:

    python3 papers/gauss-bonnet/plot_hubble_override.py
"""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG = ROOT / "inflat/out/gauss_bonnet_convergence/.mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    NullFormatter,
    PercentFormatter,
)

DATA = ROOT / "inflat/out/gauss_bonnet_convergence/alpha_1.13412e7"
FIG_DIR = Path(__file__).resolve().parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Same well-resolved comparison window as plot_convergence.py: the highest-k
# bins are affected by the finite UV cutoff and are only shaded, not compared.
K_MIN_FAIR = 1e-8
K_MAX_FRACTION = 0.7

CASES = [
    ("n128", r"$N=128$", "C3"),
    ("n128_h400", r"$N=128,\ H_{\rm ini}=H_{\rm ini}(N=400)$", "C2"),
    ("n400", r"$N=400$", "C4"),
]


def load_case(name):
    path = DATA / f"{name}.summary.npz"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            f"run: gauss_bonnet_convergence --run --config {name}"
        )
    data = np.load(path)
    return data["k"], data["p"], data["p_lin"]


def trim(k, p):
    """Drop the first and last shells, which are affected by binning/UV."""
    return k[1:-1], p[1:-1]


def interp_loglog(x, xp, fp):
    return np.exp(np.interp(np.log(x), np.log(xp), np.log(fp)))


def relative_curve(k_a, p_a, k_b, p_b, hi, n=300):
    k_a, p_a = trim(k_a, p_a)
    k_b, p_b = trim(k_b, p_b)
    lo = max(K_MIN_FAIR, k_a[0], k_b[0])
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    a = interp_loglog(grid, k_a, p_a)
    b = interp_loglog(grid, k_b, p_b)
    return grid, a / b - 1.0


def report(label, k_a, p_a, k_b, p_b, hi):
    _, rel = relative_curve(k_a, p_a, k_b, p_b, hi)
    print(
        f"{label:>28}: median |delta| = {np.median(np.abs(rel)):.3%}, "
        f"max |delta| = {np.max(np.abs(rel)):.3%}"
    )


def peak(k, p):
    k_trim, p_trim = trim(k, p)
    index = int(np.argmax(p_trim))
    return k_trim[index], p_trim[index]


def style_axes(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def style_log_xaxis(ax):
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20)
    )
    ax.xaxis.set_minor_formatter(NullFormatter())


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.8,
        }
    )

    cases = {name: load_case(name) for name, _, _ in CASES}
    k_ref, p_ref, p_lin = cases["n128"]

    k_uv = K_MAX_FRACTION * min(
        trim(k, p)[0][-1] for k, p, _ in cases.values()
    )
    k_lo = min(trim(k, p)[0][0] for k, p, _ in cases.values()) * 0.95
    # Do not let the longer N=400 tail compress the comparison window.
    k_hi = min(
        max(trim(k, p)[0][-1] for k, p, _ in cases.values()) * 1.05,
        2.0 * k_uv,
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(6.2, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )

    ax1.axvspan(k_uv, k_hi, color="0.92", zorder=0)
    ax1.axvline(k_uv, color="0.45", ls="--", lw=0.8)
    ax1.text(k_uv * 1.02, 7e-5, "UV", color="0.45", fontsize=8, va="bottom")
    ax2.axvspan(k_uv, k_hi, color="0.92", zorder=0)
    ax2.axvline(k_uv, color="0.45", ls="--", lw=0.8)

    linear_grid = np.exp(np.linspace(np.log(k_lo), np.log(k_hi), 600))
    ax1.loglog(
        linear_grid,
        interp_loglog(linear_grid, k_ref, p_lin),
        "--",
        color="0.55",
        lw=1.4,
        label="linear",
    )
    for name, label, color in CASES:
        k, p, _ = cases[name]
        k_trim, p_trim = trim(k, p)
        ax1.loglog(k_trim, p_trim, color=color, lw=1.8, label=label)

    ax1.set_ylabel(r"$\mathcal{P}_\zeta$")
    ax1.set_ylim(5e-5, 5.0)
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.92", edgecolor="none", label="UV-affected"))
    ax1.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.2,
    )
    ax1.tick_params(labelbottom=False)
    style_axes(ax1)
    style_log_xaxis(ax1)

    for name, label, color in [
        ("n128_h400", r"$N=128,\ H_{\rm ini}=H_{400}$ / $N=128$", "C2"),
        ("n400", r"$N=400$ / $N=128$", "C4"),
    ]:
        k, p, _ = cases[name]
        grid, rel = relative_curve(k, p, k_ref, p_ref, k_uv)
        ax2.semilogx(grid, rel, color=color, lw=1.5, label=label)

    ax2.axhline(0.0, color="k", ls=":", lw=1.0)
    ax2.set_xlabel(r"$k\ [\mathrm{Hz}]$")
    ax2.set_ylabel(r"$P_\zeta/P_{\zeta,N=128}-1$")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.legend(loc="upper left", frameon=False, fontsize=7.5)
    style_axes(ax2)
    style_log_xaxis(ax2)

    ax1.set_xlim(k_lo, k_hi)
    fig.align_ylabels([ax1, ax2])
    fig.savefig(FIG_DIR / "convergence-hubble-override.pdf", bbox_inches="tight")
    fig.savefig(
        FIG_DIR / "convergence-hubble-override.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"comparison window: k in [{K_MIN_FAIR:.1e}, {k_uv:.3e}] Hz")
    for name, _, _ in CASES:
        k, p, _ = cases[name]
        k_peak, p_peak = peak(k, p)
        print(f"peak {name:>10}: k = {k_peak:.3e} Hz, P = {p_peak:.3e}")

    report("n128_h400 vs n128", *cases["n128_h400"][:2], k_ref, p_ref, k_uv)
    report("n400 vs n128", *cases["n400"][:2], k_ref, p_ref, k_uv)
    report("n128_h400 vs n400", *cases["n128_h400"][:2], *cases["n400"][:2], k_uv)

    print(f"wrote {FIG_DIR / 'convergence-hubble-override.pdf'}")
    print(f"wrote {FIG_DIR / 'convergence-hubble-override.png'}")


if __name__ == "__main__":
    main()
