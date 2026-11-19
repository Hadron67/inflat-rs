#!/usr/bin/env python3
"""Analyse the fixed-dx, 2L box run `box_n400_k0_1e-9`.

The point of this run (see task.md sections 4.2.3 / 4.2.4): at fixed N=400 the
lattice zero-point energy fraction rho_vac/rho_tot is fixed by N alone
(Lambda_UV/H = 0.5513*sub*N), so it cannot be reduced by changing the box.  The
box run therefore has the *same* vacuum-energy contamination as `n400` but a
2x coarser dx and a 2x larger box.  If `box_n400` reproduces the N=200 baseline,
the n400 deficit is not a vacuum-energy effect.

Outputs a table on stdout and, with --plot, a figure next to the other
convergence figures.
"""

from pathlib import Path
import sys

import numpy as np

from plot_convergence import (
    DATA,
    BASELINE,
    XI0,
    K_MIN_FAIR,
    K_MAX_FRACTION,
    load_case,
    load_baseline,
    trim,
    interp_loglog,
)

DX = {"n64": 0.07594, "n128": 0.03797, "n300": 0.01620, "n400": 0.01215,
      "baseline": 0.02430, "box_n400_k0_1e-9": 0.02430}
NN = {"n64": 64, "n128": 128, "n300": 300, "n400": 400, "baseline": 200,
      "box_n400_k0_1e-9": 400}


def rel(k_a, p_a, k_b, p_b, lo, hi, n=400):
    # Clamp to the actual overlap: never extrapolate a bin grid.
    lo = max(lo, k_a[0], k_b[0])
    hi = min(hi, k_a[-1], k_b[-1])
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    a = interp_loglog(grid, k_a, p_a)
    b = interp_loglog(grid, k_b, p_b)
    return grid, a / b - 1.0


def stats(rel_curve):
    return np.median(np.abs(rel_curve)), np.max(np.abs(rel_curve))


def main():
    base_k, base_p, lin_k, lin_p = load_baseline()
    cases = ["n128", "n300", "n400", "n400_super500", "box_n400_k0_1e-9"]
    data = {c: load_case(c) for c in cases}

    bk, bp = trim(base_k, base_p)

    # ---- peaks -------------------------------------------------------
    print("=" * 78)
    print("peak values")
    print("=" * 78)
    print(f"{'case':<20}{'N':>5}{'dx [pc]':>10}{'k_peak [Hz]':>15}"
          f"{'P_peak':>12}{'P/P_lin':>9}")
    i = int(np.argmax(bp))
    print(f"{'baseline':<20}{200:>5}{DX['baseline']:>10.5f}{bk[i]:>15.4e}"
          f"{bp[i]:>12.4e}{bp[i] / np.interp(bk[i], lin_k, lin_p):>9.3f}")
    for c in cases:
        k, p, pl = data[c]
        k, p = trim(k, p)
        i = int(np.argmax(p))
        print(f"{c:<20}{NN.get(c, 400):>5}{DX.get(c, float('nan')):>10.5f}"
              f"{k[i]:>15.4e}{p[i]:>12.4e}{p[i] / pl[i]:>9.3f}")

    # ---- relative differences vs baseline ----------------------------
    windows = {
        "old common window (N=64)": (K_MIN_FAIR, 4.931e-8),
        "full overlap with box": (2.6e-9, K_MAX_FRACTION * 2.2052e-7),
        "peak band": (1.5e-8, 4.0e-8),
    }
    for label, (lo, hi) in windows.items():
        print()
        print("=" * 78)
        print(f"|dP/P| vs baseline over {lo:.3e} - {hi:.3e} Hz ({label})")
        print("=" * 78)
        for c in cases:
            k, p, _ = data[c]
            g, r = rel(trim(k, p)[0], trim(k, p)[1], bk, bp, lo, hi)
            med, mx = stats(r)
            print(f"  {c:<22} median {med:>7.2%}   max {mx:>7.2%}")

    # ---- is the linear (background) spectrum the same? --------------
    print()
    print("=" * 78)
    print("linear spectrum, case/baseline (background trajectory check)")
    print("=" * 78)
    grid = np.exp(np.linspace(np.log(2.6e-9), np.log(1.5e-7), 200))
    ref = interp_loglog(grid, lin_k, lin_p)
    for c in cases:
        k, p, pl = data[c]
        g = grid[(grid >= k[1]) & (grid <= 0.7 * k[-1])]
        r = interp_loglog(g, k, pl) / interp_loglog(g, lin_k, lin_p) - 1.0
        print(f"  {c:<22} median {np.median(np.abs(r)):>7.2%}   "
              f"max {np.max(np.abs(r)):>7.2%}")

    # ---- shape of the n400 deficit vs k ------------------------------
    print()
    print("=" * 78)
    print("n400 / baseline and box_n400 / baseline, by k decade")
    print("=" * 78)
    lo, hi = 2.6e-9, K_MAX_FRACTION * 2.2052e-7
    for c in ["n400", "box_n400_k0_1e-9"]:
        k, p, _ = data[c]
        g, r = rel(trim(k, p)[0], trim(k, p)[1], bk, bp, lo, hi)
        print(f"  {c}")
        for a, b in [(3e-9, 1e-8), (1e-8, 2e-8), (2e-8, 3e-8),
                     (3e-8, 6e-8), (6e-8, 1.2e-7), (1.2e-7, hi)]:
            m = (g >= a) & (g < b)
            if m.sum() < 3:
                continue
            print(f"    {a:.1e}-{b:.1e}: mean {r[m].mean():+7.2%}  "
                  f"median|.| {np.median(np.abs(r[m])):6.2%}")

    if "--plot" in sys.argv:
        make_plot(base_k, base_p, lin_k, lin_p, data)


def make_plot(base_k, base_p, lin_k, lin_p, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_convergence import (
        style_axes,
        style_log_xaxis,
        add_uv_region,
        add_uv_legend,
        FIG_DIR,
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.2, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )
    bk, bp = trim(base_k, base_p)
    k_uv = K_MAX_FRACTION * min(bk[-1], trim(data["box_n400_k0_1e-9"][0],
                                            data["box_n400_k0_1e-9"][1])[0][-1])
    k_lo = 1.15e-9
    k_hi = 2.0 * k_uv
    add_uv_region(ax1, ax2, k_uv, k_hi)

    lg = np.exp(np.linspace(np.log(k_lo), np.log(k_hi), 600))
    ax1.loglog(lg, interp_loglog(lg, lin_k, lin_p), "--", color="0.55",
               lw=1.4, label="linear")
    ax1.loglog(bk, bp, color="k", lw=1.9, label=r"$N=200$ (baseline)")
    for c, col, lab in [
        ("box_n400_k0_1e-9", "C1", r"$N=400$, $\Delta x_{\rm base}$, $2L$"),
        ("n400", "C2", r"$N=400$, $\Delta x_{\rm base}/2$"),
        ("n300", "C4", r"$N=300$"),
        ("n128", "C3", r"$N=128$"),
    ]:
        k, p, _ = data[c]
        k, p = trim(k, p)
        ax1.loglog(k, p, color=col, lw=1.8, label=lab)
    ax1.set_ylabel(r"$\mathcal{P}_\zeta$")
    ax1.set_ylim(5e-5, 2e-1)
    add_uv_legend(ax1)
    ax1.tick_params(labelbottom=False)
    style_axes(ax1)
    style_log_xaxis(ax1)

    for c, col, lab in [
        ("box_n400_k0_1e-9", "C1", r"$N=400$ ($\Delta x_{\rm base}$, $2L$)")
        ,
        ("n400", "C2", r"$N=400$"),
        ("n300", "C4", r"$N=300$"),
        ("n128", "C3", r"$N=128$"),
    ]:
        k, p, _ = data[c]
        g, r = rel(trim(k, p)[0], trim(k, p)[1], bk, bp, max(k_lo, 2.6e-9), k_hi)
        ax2.semilogx(g, 100 * r, color=col, lw=1.6, label=lab)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel(r"$\mathcal{P}/\mathcal{P}_{N=200}-1\ [\%]$")
    ax2.set_xlabel(r"$k\ [\mathrm{Hz}]$")
    ax2.set_xlim(k_lo, k_hi)
    ax2.set_ylim(-40, 40)
    ax2.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    style_axes(ax2)
    style_log_xaxis(ax2)

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"convergence-box-n400.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote", out)


if __name__ == "__main__":
    main()
