#!/usr/bin/env python3
"""Compare the three N=200 runs that share N, L, seed and initial state.

    baseline  -- production output (binary from 2025-10), dt = 2
    n200      -- current code, dt = 2
    dt1_n200  -- current code, dt = 1

All three sit on the same k grid (identical to 1e-16), so every comparison is
bin-by-bin with no interpolation.  `dt1_n200` vs `n200` isolates the time step;
`n200` vs `baseline` isolates the code version.  The result: dt matters at the
1e-5 level while the code version matters at the ~1% level (and 5-20% in the UV
tail), so `n200` -- not the production npz -- is the right reference for the
resolution scan.

Usage:
    python3 analyze_dt1_n200.py [--plot]
"""

import sys

import numpy as np

from plot_convergence import BASELINE, DATA, FIG_DIR, XI0, style_axes, style_log_xaxis

CASES = ("baseline", "n200", "dt1_n200")
BANDS = [
    (2.0e-9, 5.0e-9),
    (5.0e-9, 1.0e-8),
    (1.0e-8, 2.0e-8),
    (2.0e-8, 3.0e-8),
    (3.0e-8, 6.0e-8),
    (6.0e-8, 1.2e-7),
    (1.2e-7, 2.3e-7),
]


def load(case):
    if case == "baseline":
        b = np.load(BASELINE)
        row = int(np.where(np.abs(b["params"] - XI0) < 1.0)[0][0])
        return dict(k=np.asarray(b["lat_k"]), p=np.asarray(b["lat_spectrums"])[row],
                    pl=None, dt=2.0)
    d = np.load(DATA / f"{case}.summary.npz")
    return dict(k=d["k"], p=d["p"], pl=d["p_lin"], dt=float(d["dt"][0]), d=d)


def band_table(label, r, k):
    print(f"\n{label}:")
    print(f"  {'band [Hz]':<24}{'bins':>5}{'mean':>9}{'median|.|':>11}{'max|.|':>9}")
    for lo, hi in BANDS:
        m = (k >= lo) & (k < hi)
        if m.sum() < 2:
            continue
        print(f"  {lo:.1e} - {hi:.1e}   {m.sum():>5}"
              f"{100 * r[m].mean():>+8.2f}%{100 * np.median(np.abs(r[m])):>10.2f}%"
              f"{100 * np.max(np.abs(r[m])):>8.2f}%")


def main():
    D = {c: load(c) for c in CASES}
    kb = D["baseline"]["k"]
    k = D["n200"]["k"]
    if k.shape != kb.shape or np.max(np.abs(k / kb - 1)) > 1e-12:
        raise SystemExit("the three N=200 runs are expected to share one k grid")

    print("=" * 74)
    print("three N=200 runs, identical k grid, no interpolation")
    print("=" * 74)
    print(f"{'case':<12}{'dt':>4}{'k_peak [Hz]':>16}{'P_peak':>14}")
    for c in CASES:
        i = int(np.argmax(D[c]["p"]))
        print(f"{c:<12}{D[c]['dt']:>4.0f}{D[c]['k'][i]:>16.6e}{D[c]['p'][i]:>14.6e}")

    r_dt = D["dt1_n200"]["p"] / D["n200"]["p"] - 1.0
    r_cv = D["baseline"]["p"] / D["n200"]["p"] - 1.0
    m = (k > 1e-8) & (k < 0.7 * k[-1])
    for lab, r in (("dt=1 vs dt=2   (same code)", r_dt),
                   ("baseline vs n200 (code version)", r_cv)):
        print(f"\n{lab}")
        print(f"  at the peak   {100 * r[int(np.argmax(D['n200']['p']))]:+.4f}%")
        print(f"  median|.|     {100 * np.median(np.abs(r)):.4f}%")
        print(f"  max|.|        {100 * np.max(np.abs(r)):.4f}%")
        print(f"  fair window [1e-8, 0.7 k_max]: median|.| "
              f"{100 * np.median(np.abs(r[m])):.2f}%")

    band_table("bin-by-bin: dt=1 vs dt=2", r_dt, k)
    band_table("bin-by-bin: baseline vs n200", r_cv, k)

    print("\nzeta_lin (spectrums[-1]):")
    z2 = D["n200"]["d"]["spectrums"][-1]
    z1 = D["dt1_n200"]["d"]["spectrums"][-1]
    print(f"  n200 peak {np.max(z2):.6e}, dt1_n200 peak {np.max(z1):.6e}")
    print(f"  dt=1 vs dt=2: median|.| {100 * np.median(np.abs(z1 / z2 - 1)):.6f}%, "
          f"max|.| {100 * np.max(np.abs(z1 / z2 - 1)):.6f}%")

    if "--plot" in sys.argv:
        make_plot(D)


def make_plot(D):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = D["n200"]["k"]
    t = slice(1, -1)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.2, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )
    ax1.loglog(k[t], D["baseline"]["p"][t], color="k", lw=2.2,
               label=r"baseline (2025-10 code, $\Delta t=2$)")
    ax1.loglog(k[t], D["n200"]["p"][t], color="C1", lw=1.4,
               label=r"current code, $\Delta t=2$")
    ax1.loglog(k[t], D["dt1_n200"]["p"][t], color="C3", lw=1.0, ls=":",
               label=r"current code, $\Delta t=1$")
    ax1.set_ylabel(r"$\mathcal{P}_\zeta$")
    ax1.set_ylim(5e-5, 2e-1)
    ax1.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01))
    ax1.tick_params(labelbottom=False)
    style_axes(ax1)
    style_log_xaxis(ax1)

    ax2.semilogx(k[t], 100 * (D["dt1_n200"]["p"][t] / D["n200"]["p"][t] - 1),
                 color="C3", lw=1.8, label=r"$\Delta t=1$ vs $\Delta t=2$ (same code)")
    ax2.semilogx(k[t], 100 * (D["baseline"]["p"][t] / D["n200"]["p"][t] - 1),
                 color="k", lw=1.6, label="baseline vs current code")
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set_ylabel(r"$\mathcal{P}/\mathcal{P}_{N=200}^{\rm current}-1\ [\%]$")
    ax2.set_xlabel(r"$k\ [\mathrm{Hz}]$")
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    style_axes(ax2)
    style_log_xaxis(ax2)

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"convergence-dt-n200.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote", out)


if __name__ == "__main__":
    main()
