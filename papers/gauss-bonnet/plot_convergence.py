#!/usr/bin/env python3
"""Plot the lattice convergence tests for alpha = 1.13412e7.

Two figures are produced:

  * convergence-resolution: N=64, N=128, N=300, N=400 and the existing N=200
    baseline.
  * convergence-dt:         N=64 with dt=1 and dt=2.

SIGW is intentionally not plotted.  The current `sigw_2_spectrum` truncates
modes outside the sampled k range; for the small N=64 box this truncation is
large and the resulting Omega_GW is not a meaningful convergence diagnostic.
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
BASELINE = ROOT / "inflat/out/gauss_bonnet.bak/case1-final.output.npz"
FIG_DIR = Path(__file__).resolve().parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

XI0 = 1.13412e7

# Well-resolved comparison window.  The highest-k lattice bins are affected by
# the finite UV cutoff, especially for N=64, and are shown only in the upper
# panels (shaded) but excluded from the relative-difference curves.
K_MIN_FAIR = 1e-8
K_MAX_FRACTION = 0.7


def load_case(name):
    data = np.load(DATA / f"{name}.summary.npz")
    return data["k"], data["p"], data["p_lin"]


def load_baseline():
    data = np.load(BASELINE)
    # Use an absolute tolerance: the grid spacing in alpha is 100, and
    # np.isclose's default relative tolerance would also select 1.13411e7.
    matches = np.where(np.abs(data["params"] - XI0) < 1.0)[0]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one baseline row, found {matches}")
    row = matches[0]
    return (
        data["lat_k"],
        data["lat_spectrums"][row],
        data["linear_k"],
        data["linear_spectrums"][row],
    )


def trim(k, p):
    """Drop the first and last shells, which are affected by binning/UV."""
    return k[1:-1], p[1:-1]


def interp_loglog(x, xp, fp):
    return np.exp(np.interp(np.log(x), np.log(xp), np.log(fp)))


def common_hi(*cases):
    return K_MAX_FRACTION * min(trim(k, p)[0][-1] for k, p in cases)


def relative_curve(k_a, p_a, k_b, p_b, n=300, hi=None):
    k_a, p_a = trim(k_a, p_a)
    k_b, p_b = trim(k_b, p_b)
    lo = max(K_MIN_FAIR, k_a[0], k_b[0])
    if hi is None:
        hi = K_MAX_FRACTION * min(k_a[-1], k_b[-1])
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    a = interp_loglog(grid, k_a, p_a)
    b = interp_loglog(grid, k_b, p_b)
    return grid, a / b - 1.0


def report(label, k_a, p_a, k_b, p_b, hi=None):
    _, rel = relative_curve(k_a, p_a, k_b, p_b, hi=hi)
    print(
        f"{label:>22}: median |delta| = {np.median(np.abs(rel)):.3%}, "
        f"max |delta| = {np.max(np.abs(rel)):.3%}"
    )


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


def add_uv_region(ax_top, ax_bottom, k_uv, k_hi):
    ax_top.axvspan(k_uv, k_hi, color="0.92", zorder=0)
    ax_top.axvline(k_uv, color="0.45", ls="--", lw=0.8)
    ax_top.text(k_uv * 1.02, 7e-5, "UV", color="0.45", fontsize=8, va="bottom")
    ax_bottom.axvspan(k_uv, k_hi, color="0.92", zorder=0)
    ax_bottom.axvline(k_uv, color="0.45", ls="--", lw=0.8)


def add_uv_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.92", edgecolor="none", label="UV-affected"))
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.2,
    )


def make_resolution_plot(base_k, base_p, lin_k, lin_p, n64, n128, n300, n400):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(6.2, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )

    base_k_t, base_p_t = trim(base_k, base_p)
    n64_k, n64_p = trim(n64[0], n64[1])
    n128_k, n128_p = trim(n128[0], n128[1])
    n300_k, n300_p = trim(n300[0], n300[1])
    n400_k, n400_p = trim(n400[0], n400[1])

    k_lo = min(base_k_t[0], n64_k[0], n128_k[0], n300_k[0], n400_k[0]) * 0.95
    k_hi_raw = (
        max(base_k_t[-1], n64_k[-1], n128_k[-1], n300_k[-1], n400_k[-1]) * 1.05
    )
    k_uv = common_hi(
        (base_k, base_p),
        (n64[0], n64[1]),
        (n128[0], n128[1]),
        (n300[0], n300[1]),
        (n400[0], n400[1]),
    )
    # Show the peak and the beginning of the UV region, but do not let the
    # much longer N=400 tail compress the comparison window.
    k_hi = min(k_hi_raw, 2.0 * k_uv)

    add_uv_region(ax1, ax2, k_uv, k_hi)

    linear_grid = np.exp(np.linspace(np.log(k_lo), np.log(k_hi), 600))
    ax1.loglog(
        linear_grid,
        interp_loglog(linear_grid, lin_k, lin_p),
        "--",
        color="0.55",
        lw=1.4,
        label="linear",
    )
    ax1.loglog(base_k_t, base_p_t, color="k", lw=1.8, label=r"$N=200$ (baseline)")
    ax1.loglog(n400_k, n400_p, color="C2", lw=1.8, label=r"$N=400$")
    ax1.loglog(n300_k, n300_p, color="C4", lw=1.8, label=r"$N=300$")
    ax1.loglog(n128_k, n128_p, color="C3", lw=1.8, label=r"$N=128$")
    ax1.loglog(n64_k, n64_p, color="C0", lw=1.8, label=r"$N=64$")

    ax1.set_ylabel(r"$\mathcal{P}_\zeta$")
    ax1.set_ylim(5e-5, 2e-1)
    add_uv_legend(ax1)
    ax1.tick_params(labelbottom=False)
    style_axes(ax1)
    style_log_xaxis(ax1)

    for data, color, label in [
        (n400, "C2", r"$N=400$ / baseline"),
        (n300, "C4", r"$N=300$ / baseline"),
        (n128, "C3", r"$N=128$ / baseline"),
        (n64, "C0", r"$N=64$ / baseline"),
    ]:
        grid, rel = relative_curve(data[0], data[1], base_k, base_p, hi=k_uv)
        ax2.semilogx(grid, rel, color=color, lw=1.5, label=label)

    ax2.axhline(0.0, color="k", ls=":", lw=1.0)
    ax2.set_xlabel(r"$k\ [\mathrm{Hz}]$")
    ax2.set_ylabel(r"$P_\zeta/P_{\zeta,\rm base}-1$")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.legend(loc="upper left", frameon=False, ncol=2, fontsize=7.5)
    style_axes(ax2)
    style_log_xaxis(ax2)

    ax1.set_xlim(k_lo, k_hi)
    fig.align_ylabels([ax1, ax2])
    fig.savefig(FIG_DIR / "convergence-resolution.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "convergence-resolution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_dt_plot(dt2, dt1):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(6.2, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )

    dt2_k, dt2_p = trim(dt2[0], dt2[1])
    dt1_k, dt1_p = trim(dt1[0], dt1[1])
    _, dt2_lin = trim(dt2[0], dt2[2])

    k_lo = min(dt1_k[0], dt2_k[0]) * 0.95
    k_hi = max(dt1_k[-1], dt2_k[-1]) * 1.05
    k_uv = common_hi((dt1[0], dt1[1]), (dt2[0], dt2[1]))

    add_uv_region(ax1, ax2, k_uv, k_hi)

    linear_grid = np.exp(np.linspace(np.log(k_lo), np.log(k_hi), 600))
    ax1.loglog(
        linear_grid,
        interp_loglog(linear_grid, dt2_k, dt2_lin),
        "--",
        color="0.55",
        lw=1.4,
        label="linear",
    )
    ax1.loglog(dt2_k, dt2_p, color="C0", lw=1.8, label=r"$N=64,\ \Delta t=2$")
    ax1.loglog(dt1_k, dt1_p, color="C2", lw=1.8, label=r"$N=64,\ \Delta t=1$")

    ax1.set_ylabel(r"$\mathcal{P}_\zeta$")
    ax1.set_ylim(5e-5, 2e-1)
    add_uv_legend(ax1)
    ax1.tick_params(labelbottom=False)
    style_axes(ax1)
    style_log_xaxis(ax1)

    grid, rel = relative_curve(dt1[0], dt1[1], dt2[0], dt2[1], hi=k_uv)
    ax2.semilogx(grid, rel, color="C2", lw=1.5)
    ax2.axhline(0.0, color="k", ls=":", lw=1.0)
    ax2.set_xlabel(r"$k\ [\mathrm{Hz}]$")
    ax2.set_ylabel(r"$P_{\Delta t=1}/P_{\Delta t=2}-1$")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    style_axes(ax2)
    style_log_xaxis(ax2)

    ax1.set_xlim(k_lo, k_hi)
    fig.align_ylabels([ax1, ax2])
    fig.savefig(FIG_DIR / "convergence-dt.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "convergence-dt.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


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

    base_k, base_p, lin_k, lin_p = load_baseline()
    n64 = load_case("n64")
    n128 = load_case("n128")
    n300 = load_case("n300")
    n400 = load_case("n400")
    dt1 = load_case("dt1_n64")

    make_resolution_plot(base_k, base_p, lin_k, lin_p, n64, n128, n300, n400)
    make_dt_plot(n64, dt1)

    hi_res = common_hi(
        (base_k, base_p),
        (n64[0], n64[1]),
        (n128[0], n128[1]),
        (n300[0], n300[1]),
        (n400[0], n400[1]),
    )
    hi_dt = common_hi((n64[0], n64[1]), (dt1[0], dt1[1]))

    print(f"resolution window: k in [{K_MIN_FAIR:.1e}, {hi_res:.3e}] Hz")
    print(f"dt window:         k in [{K_MIN_FAIR:.1e}, {hi_dt:.3e}] Hz")
    report("N=64 vs baseline", n64[0], n64[1], base_k, base_p, hi=hi_res)
    report("N=128 vs baseline", n128[0], n128[1], base_k, base_p, hi=hi_res)
    report("N=300 vs baseline", n300[0], n300[1], base_k, base_p, hi=hi_res)
    report("N=400 vs baseline", n400[0], n400[1], base_k, base_p, hi=hi_res)
    report("N=64 dt=1 vs dt=2", dt1[0], dt1[1], n64[0], n64[1], hi=hi_dt)

    print(f"wrote {FIG_DIR / 'convergence-resolution.pdf'}")
    print(f"wrote {FIG_DIR / 'convergence-dt.pdf'}")


if __name__ == "__main__":
    main()
