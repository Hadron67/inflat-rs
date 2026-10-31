#!/usr/bin/env python3
"""Plot the mid-run spectra stored in ``LatticeOutputData::spectrums``.

For a given convergence case this checks whether the curvature power spectrum
has frozen by the end of the run.  The stored snapshots are

    (ln a, P_phi(k) * (H / <v_phi>)^2)

i.e. a linearised estimate of P_zeta at that time.  The final, non-linear
delta-N spectrum is also shown for reference.

The snapshots are in simulation units.  ``aH`` is converted to Hz with a
constant factor fixed by the start/end horizon conditions of the run.

Usage:
    python3 papers/gauss-bonnet/plot_evolution.py [case ...]

With no arguments it plots ``n400``.
"""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG = ROOT / "inflat/out/gauss_bonnet_convergence/.mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    NullFormatter,
    PercentFormatter,
)

DATA = ROOT / "inflat/out/gauss_bonnet_convergence/alpha_1.13412e7"
FIG_DIR = Path(__file__).resolve().parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Well-resolved window used by the convergence figures.
K_MIN_FAIR = 1e-8
K_MAX_FRACTION = 0.7

# Modes whose growth history is highlighted in the middle panel.
TRACK_K = [1.1e-8, 1.6e-8, 2.4e-8, 3.5e-8, 4.9e-8]


def trim(k, p):
    return k[1:-1], p[1:-1]


def nearest_bin(k, target):
    return int(np.argmin(np.abs(k - target)))


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


def load(name):
    d = np.load(DATA / f"{name}.summary.npz")
    k = d["spectrum_k"]
    a = d["measurable_a"]
    v_a = d["measurable_v_a"]
    k0 = float(d["k0"][0])
    sub = float(d["subhorizon_tolerance"][0])
    sup = float(d["superhorizon_tolerance"][0])

    # aH_Hz = v_a * C.  The start state has aH >= k0/sub and the end state has
    # aH >= k_max*sup, so these two estimates bracket C.
    c_start = (k0 / sub) / v_a[0]
    c_end = (k[-1] * sup) / float(d["final_v_a"][0])
    c = 0.5 * (c_start + c_end)

    return {
        "name": name,
        "n": int(d["N"][0]),
        "k": k,
        "p": d["p"],
        "p_lin": d["p_lin"],
        "ln_a": d["spectrum_ln_a"],
        "spectrums": d["spectrums"],
        "a": a,
        "v_a": v_a,
        "v_phi": d["measurable_v_phi"],
        "final_ln_a": float(np.log(d["final_a"][0])),
        "c": c,
        "c_spread": abs(c_end - c_start) / c,
    }


def make_plot(case):
    k = case["k"]
    p = case["p"]
    p_lin = case["p_lin"]
    ln_a = case["ln_a"]
    spectrums = case["spectrums"]
    a = case["a"]
    v_a = case["v_a"]
    v_phi = case["v_phi"]
    c = case["c"]

    k_t, p_t = trim(k, p)
    k_lo = k_t[0] * 0.95
    k_hi = k[-1] * 1.05
    k_uv = K_MAX_FRACTION * k_t[-1]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(11.5, 4.0),
        gridspec_kw={"width_ratios": [1.6, 1.05, 0.75], "wspace": 0.3},
    )

    # ---- left: spectrum snapshots -------------------------------------
    ax1.axvspan(k_uv, k_hi, color="0.92", zorder=0)
    ax1.axvline(K_MIN_FAIR, color="0.45", ls=":", lw=0.8)
    ax1.axvline(k_uv, color="0.45", ls="--", lw=0.8)

    norm = plt.Normalize(ln_a[0], ln_a[-1])
    cmap = cm.viridis
    for la, spec in zip(ln_a, spectrums):
        ax1.loglog(
            k[1:-1],
            spec[1:-1],
            color=cmap(norm(la)),
            lw=1.2,
            alpha=0.9,
        )

    ax1.loglog(
        k_t, np.interp(k_t, k, p_lin), "--", color="0.55", lw=1.4, label="linear"
    )
    ax1.loglog(k_t, p_t, color="k", lw=2.0, label=r"final $\delta N$")

    ax1.set_xlabel(r"$k\ [\mathrm{Hz}]$")
    ax1.set_ylabel(r"$\mathcal{P}_\zeta$ (mid-run estimate)")
    ax1.set_xlim(k_lo, k_hi)
    ax1.set_ylim(1e-4, 3e-1)
    style_axes(ax1)
    style_log_xaxis(ax1)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax1, pad=0.015, fraction=0.045)
    cbar.set_label(r"$\ln a$")

    handles = [
        plt.Line2D([], [], color="k", lw=2.0),
        plt.Line2D([], [], color="0.55", lw=1.4, ls="--"),
    ]
    ax1.legend(handles, [r"final $\delta N$", "linear"],
               loc="lower left", frameon=False, fontsize=8)
    ax1.text(k_uv * 1.02, 2.5e-4, "UV", color="0.45", fontsize=8, va="bottom")

    # ---- middle: growth relative to the last snapshot ------------------
    last = spectrums[-1]
    ratio_all = spectrums / last[None, :]

    ax2.axhspan(0.95, 1.05, color="0.92", zorder=0)
    ax2.axhline(1.0, color="0.4", ls=":", lw=0.9)

    me_ln_a = np.log(a)
    crossings = []
    for j, k_target in enumerate(TRACK_K):
        idx = nearest_bin(k, k_target)
        actual_k = k[idx]
        color = f"C{j}"
        ratio = ratio_all[:, idx]
        ax2.plot(
            ln_a,
            ratio,
            "o-",
            color=color,
            lw=1.4,
            ms=3.5,
            label=r"$k=%.1f\times10^{-8}$ Hz" % (actual_k / 1e-8),
        )

        # Horizon crossing: aH = k, with aH_Hz = v_a * C.
        crossing = np.nan
        v_needed = actual_k / c
        if v_a[0] <= v_needed <= v_a[-1]:
            crossing = float(np.interp(v_needed, v_a, me_ln_a))
        crossings.append((actual_k, crossing, idx))

    ax2.set_xlabel(r"$\ln a$ (late evolution)")
    ax2.set_ylabel(r"$\mathcal{P}_\zeta(t)/\mathcal{P}_\zeta(t_{\rm end})$")
    ax2.set_ylim(0.6, 1.15)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_xlim(4.0, ln_a[-1] + 0.4)
    style_axes(ax2)
    ax2.legend(loc="lower left", frameon=False, fontsize=7.5)

    # ---- right: estimator normalization --------------------------------
    factor = (v_a / a) / v_phi
    factor_snap = np.interp(ln_a, me_ln_a, factor)
    ax3.semilogy(ln_a[1:], factor_snap[1:] / factor_snap[-1],
                 "o-", color="0.25", lw=1.4, ms=3.5)
    ax3.axhline(1.0, color="0.4", ls=":", lw=0.9)
    ax3.set_xlabel(r"$\ln a$")
    ax3.set_ylabel(r"$(H/\langle\dot\phi\rangle)\,/\,{\rm final}$")
    ax3.set_xlim(ln_a[1], ln_a[-1])
    style_axes(ax3)
    ax3.set_title("estimator factor", fontsize=8.5)

    fig.suptitle(
        r"$N=%d$, $\alpha=1.13412\times10^{7}$: mid-run spectra"
        % case["n"],
        fontsize=10,
        y=0.99,
    )
    fig.savefig(FIG_DIR / f"convergence-evolution-{case['name']}.pdf",
                bbox_inches="tight")
    fig.savefig(FIG_DIR / f"convergence-evolution-{case['name']}.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[{case['name']}] N={case['n']} final ln a={case['final_ln_a']:.3f}, "
          f"aH calibration spread {case['c_spread']:.2%}")
    print("  k [Hz]    horizon-exit ln a   k/(aH) at end   "
          "P(t)/P(end) at ln a-2   last-step change")
    for actual_k, crossing, idx in crossings:
        ratio = ratio_all[:, idx]
        target = ln_a[-1] - 2.0
        r_at = float(np.interp(target, ln_a, ratio))
        print(f"  {actual_k:.3e}  {crossing:10.3f}      "
              f"{actual_k / (v_a[-1] * c):11.2e}     {r_at:8.3f}          "
              f"{ratio[-1] - ratio[-2]:+.1%}")


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8,
            "lines.linewidth": 1.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.8,
        }
    )

    names = sys.argv[1:] or ["n400"]
    for name in names:
        make_plot(load(name))
        print(f"  wrote {FIG_DIR / f'convergence-evolution-{name}.pdf'}")


if __name__ == "__main__":
    main()
