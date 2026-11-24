#!/usr/bin/env python3
"""Seed control at N=400: `seed11_n400` vs `n400` (seed=10).

Everything except the random seed is identical (N=400, L=4.86 pc,
dx=0.01215 pc, dt=2, sub=1.3, super=200, same binary), so the two runs differ
only in the TWA mode amplitudes.  This is the direct test of whether the ~10%
N=400 deficit relative to `n200` (task.md sections 4.2 / 4.2.6 item 2) is a
random-realisation effect or a systematic one.

The comparison uses the same k grid (both runs have N=400, so the shell
binning is identical) and, for the seed scatter, the per-shell cosmic variance

    sigma(P_l) / P_l = 1 / sqrt(M_l),
    sigma(P_l^a / P_l^b) = sqrt(2 / M_l),

where M_l is the number of lattice modes with round(|n|) = l, counted exactly
for N=400.  A difference that is consistent with sqrt(2/M_l) is random; one
that is much larger is systematic.

Usage:
    python3 analyze_seed11_n400.py [--plot]
"""

import sys

import numpy as np

from plot_convergence import (
    K_MAX_FRACTION,
    K_MIN_FAIR,
    interp_loglog,
    load_case,
    trim,
)

NN = 400

# Reference windows used throughout task.md.
COMMON_HI = 4.931e-8          # 0.7 * k_max(N=64)
PEAK_BAND = (1.5e-8, 4.0e-8)

BANDS = [
    (3.0e-9, 1.0e-8),
    (1.0e-8, 2.0e-8),
    (2.0e-8, 3.0e-8),
    (3.0e-8, 6.0e-8),
    (6.0e-8, 1.2e-7),
    (1.2e-7, 2.3e-7),
]


def mode_counts(n):
    """Number of modes in shell l = round(sqrt(nx^2+ny^2+nz^2)) for an n^3 box.

    Signs are irrelevant (only squares enter), so this counts triples over the
    signed FFT mode range.  Reproduces the 1250 modes of the N=400 peak shell.
    """
    half = n // 2
    coords = np.arange(-half, half)
    nz2 = coords.astype(np.int64) ** 2
    counts = {}
    for nx in coords:
        r = np.sqrt(nx * nx + np.add.outer(coords.astype(np.int64) ** 2, nz2))
        l = np.rint(r).astype(np.int64).ravel()
        vals, cnt = np.unique(l, return_counts=True)
        for v, c in zip(vals, cnt):
            counts[int(v)] = counts.get(int(v), 0) + int(c)
    counts.pop(0, None)
    return counts


def rel(k_a, p_a, k_b, p_b, lo, hi, n=400):
    lo = max(lo, k_a[0], k_b[0])
    hi = min(hi, k_a[-1], k_b[-1])
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    a = interp_loglog(grid, k_a, p_a)
    b = interp_loglog(grid, k_b, p_b)
    return grid, a / b - 1.0


def main():
    cases = ["n128", "n200", "n300", "n400", "seed11_n400"]
    data = {c: load_case(c) for c in cases}

    k10, p10, pl10 = data["n400"]
    k11, p11, pl11 = data["seed11_n400"]
    k20, p20, pl20 = data["n200"]

    print("=" * 78)
    print("peak values")
    print("=" * 78)
    print(f"{'case':<14}{'k_peak [Hz]':>16}{'P_peak':>13}{'P/P_lin':>9}"
          f"{'peak bin l':>11}")
    l_of = {}
    for c in cases:
        k, p, pl = data[c]
        kt, pt = trim(k, p)
        plt_ = trim(k, pl)[1]
        i = int(np.argmax(pt))
        l_of[c] = i + 2  # bins are sorted; shell 1 is the first, dropped by trim
        print(f"{c:<14}{kt[i]:>16.6e}{pt[i]:>13.6e}{pt[i] / plt_[i]:>9.3f}"
              f"{l_of[c]:>11}")

    # ---------------------------------------------------------------
    # 1. seed11 vs seed10: identical grid, no interpolation
    # ---------------------------------------------------------------
    if k10.shape != k11.shape or np.max(np.abs(k10 / k11 - 1)) > 1e-10:
        raise SystemExit("n400 and seed11_n400 must share the same k grid")

    r_seed = p11 / p10 - 1.0
    counts = mode_counts(NN)
    l_grid = np.arange(1, len(k10) + 1)
    m_l = np.array([counts.get(int(l), 0) for l in l_grid], dtype=float)

    print()
    print("=" * 78)
    print("seed=11 vs seed=10 (N=400), bin by bin, identical k grid")
    print("=" * 78)
    print(f"  whole overlap            median {np.median(np.abs(r_seed)):>7.3%}"
          f"   max {np.max(np.abs(r_seed)):>7.3%}")
    for lo, hi in [(K_MIN_FAIR, COMMON_HI), PEAK_BAND]:
        m = (k10 >= lo) & (k10 <= hi)
        print(f"  {lo:.1e}-{hi:.1e} Hz      median {np.median(np.abs(r_seed[m])):>7.3%}"
              f"   max {np.max(np.abs(r_seed[m])):>7.3%}")

    # per-bin significance against cosmic variance
    sig = np.sqrt(2.0 / m_l)
    z = r_seed / sig
    lpk = l_of["n400"]
    print(f"\n  cosmic variance reference (1 sigma of a two-realisation ratio):")
    for l in (3, 5, 10, 12, 20, 50, 100):
        if l <= len(m_l) and m_l[l - 1] > 0:
            print(f"    l={l:<4d} M={int(m_l[l - 1]):>8d}   sqrt(2/M) = "
                  f"{100 * np.sqrt(2.0 / m_l[l - 1]):>6.2f}%")
    print(f"  normalised residual r/sigma: median {np.median(np.abs(z)):.2f}, "
          f"95th pct {np.percentile(np.abs(z), 95):.2f}, max {np.max(np.abs(z)):.2f}")
    print(f"  (a pure pair of independent realisations gives median |z| = 0.67, "
          f"95th pct = 1.96)")

    # ---------------------------------------------------------------
    # 2. both N=400 runs against the N=200 reference
    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("both N=400 realisations against the N=200 reference (`n200`)")
    print("=" * 78)
    print(f"{'window':<26}{'case':<14}{'median |.|':>12}{'max |.|':>10}")
    for label, (lo, hi) in [
        (f"common window", (K_MIN_FAIR, COMMON_HI)),
        (f"peak band", PEAK_BAND),
        (f"full overlap", (2.6e-9, K_MAX_FRACTION * 2.2052e-7)),
    ]:
        for c in ["n400", "seed11_n400"]:
            k, p, _ = data[c]
            g, r = rel(trim(k, p)[0], trim(k, p)[1], *trim(k20, p20), lo, hi)
            print(f"{label:<26}{c:<14}{np.median(np.abs(r)):>11.2%}"
                  f"{np.max(np.abs(r)):>10.2%}")
        # also n300/n128 for context, common window only
        if label == "common window":
            for c in ["n300", "n128"]:
                k, p, _ = data[c]
                g, r = rel(trim(k, p)[0], trim(k, p)[1], *trim(k20, p20), lo, hi)
                print(f"{'':<26}{c:<14}{np.median(np.abs(r)):>11.2%}"
                      f"{np.max(np.abs(r)):>10.2%}")

    # ---------------------------------------------------------------
    # 3. how much of the N=400 deficit does a second seed remove?
    # ---------------------------------------------------------------
    i10 = int(np.argmax(trim(k10, p10)[1]))
    i11 = int(np.argmax(trim(k11, p11)[1]))
    i20 = int(np.argmax(trim(k20, p20)[1]))
    k20t, p20t = trim(k20, p20)
    pk20 = p20t[i20]
    print()
    print("=" * 78)
    print("peak amplitude relative to n200")
    print("=" * 78)
    print(f"  n200           P_peak = {pk20:.6e}")
    print(f"  n400  (seed10) P_peak = {trim(k10, p10)[1][i10]:.6e}"
          f"   ({trim(k10, p10)[1][i10] / pk20 - 1:+.2%})"
          f"  at k = {trim(k10, p10)[0][i10]:.4e} Hz")
    print(f"  seed11_n400    P_peak = {trim(k11, p11)[1][i11]:.6e}"
          f"   ({trim(k11, p11)[1][i11] / pk20 - 1:+.2%})"
          f"  at k = {trim(k11, p11)[0][i11]:.4e} Hz")
    print(f"  seed11 vs seed10 at the common peak shell l={l_of['n400']}: "
          f"{100 * r_seed[l_of['n400'] - 1]:+.2f}%")

    # band table, seed11 vs n400
    print()
    print("=" * 78)
    print("seed11_n400 / n400 - 1, by k band")
    print("=" * 78)
    g, r = rel(trim(k11, p11)[0], trim(k11, p11)[1],
               trim(k10, p10)[0], trim(k10, p10)[1], 2.6e-9,
               K_MAX_FRACTION * 2.2052e-7)
    for lo, hi in BANDS:
        m = (g >= lo) & (g < hi)
        if m.sum() < 3:
            continue
        print(f"  {lo:.1e}-{hi:.1e}: mean {r[m].mean():+7.2%}  "
              f"median|.| {np.median(np.abs(r[m])):6.2%}")

    # linear spectra (background trajectory) must be identical up to noise
    print()
    print("=" * 78)
    print("linear spectrum ratio seed11/n400 (background + linear growth only)")
    print("=" * 78)
    m = (k10 >= 3e-9) & (k10 <= 1.5e-7)
    print(f"  median {np.median(np.abs(pl11[m] / pl10[m] - 1)):.2%}   "
          f"max {np.max(np.abs(pl11[m] / pl10[m] - 1)):.2%}")

    if "--plot" in sys.argv:
        make_plot(data)


def make_plot(data):
    import matplotlib.pyplot as plt
    from plot_convergence import (
        add_uv_legend,
        add_uv_region,
        style_axes,
        style_log_xaxis,
        FIG_DIR,
    )
    from plot_convergence import load_baseline

    base_k, base_p, lin_k, lin_p = load_baseline()
    bk, bp = trim(base_k, base_p)
    k20, p20, _ = data["n200"]
    k20t, p20t = trim(k20, p20)
    k10, p10, _ = data["n400"]
    k11, p11, _ = data["seed11_n400"]
    k10t, p10t = trim(k10, p10)
    k11t, p11t = trim(k11, p11)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.2, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )
    k_lo = 1.15e-9
    k_uv = K_MAX_FRACTION * min(k20t[-1], k10t[-1])
    k_hi = 2.0 * K_MAX_FRACTION * min(k20t[-1], k10t[-1])
    add_uv_region(ax1, ax2, k_uv, k_hi)

    lg = np.exp(np.linspace(np.log(k_lo), np.log(k_hi), 600))
    ax1.loglog(lg, interp_loglog(lg, lin_k, lin_p), "--", color="0.55",
               lw=1.4, label="linear")
    ax1.loglog(k20t, p20t, color="k", lw=1.9, label=r"$N=200$ ($n200$)")
    ax1.loglog(k10t, p10t, color="C2", lw=1.6, label=r"$N=400$, seed 10")
    ax1.loglog(k11t, p11t, color="C3", lw=1.6, label=r"$N=400$, seed 11")
    ax1.set_ylabel(r"$\mathcal{P}_\zeta$")
    ax1.set_ylim(5e-5, 2e-1)
    add_uv_legend(ax1)
    ax1.tick_params(labelbottom=False)
    style_axes(ax1)
    style_log_xaxis(ax1)

    lo, hi = max(k_lo, 2.6e-9), k_hi
    g, r10 = rel(k10t, p10t, k20t, p20t, lo, hi)
    _, r11 = rel(k11t, p11t, k20t, p20t, lo, hi)
    ax2.semilogx(g, 100 * r10, color="C2", lw=1.6, label="seed 10 / N=200")
    ax2.semilogx(g, 100 * r11, color="C3", lw=1.6, label="seed 11 / N=200")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel(r"$\mathcal{P}/\mathcal{P}_{N=200}-1\ [\%]$")
    ax2.set_xlabel(r"$k\ [\mathrm{Hz}]$")
    ax2.set_xlim(k_lo, k_hi)
    ax2.set_ylim(-40, 40)
    ax2.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    style_axes(ax2)
    style_log_xaxis(ax2)

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"convergence-seed-n400.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote", out)


if __name__ == "__main__":
    main()
