#!/usr/bin/env python3
"""Compare the `spectrums` snapshots -- the curvature spectrum extracted from the
lattice through the *linear* relation

    zeta_lin(k) = -H dphi_k / <phidot>     =>     P_lin = (H/<phidot>)^2 P_dphi

-- against the final delta-N spectrum `p`, across dt and across N.

Pitfalls this script handles:
  * `spectrums[0]` uses a^2 instead of (H/<phidot>)^2, so it is skipped;
  * zeta_lin is NOT frozen (it drifts +23% over the last e-fold at N=128), so
    runs are compared at matched (k, ln a), not at their final snapshots;
  * end_n differs between resolutions (ln N), so snapshot[-1] is at a different
    time for each run.
"""

import sys

import numpy as np

from plot_convergence import DATA, FIG_DIR, style_axes, style_log_xaxis

CASES = ["n128", "n300", "n400", "dt1_n200", "dt4_n128", "dt8_n128"]
REF = "dt1_n200"
K_WINDOW = (1.5e-8, 6.0e-8)
LA_WINDOW = (7.0, 9.5)


def load(case):
    d = np.load(DATA / f"{case}.summary.npz")
    return dict(
        k=d["k"],
        la=np.asarray(d["spectrum_ln_a"]),
        s=np.asarray(d["spectrums"]),
        dn=d["p"],
        N=int(d["N"][0]),
        dt=float(d["dt"][0]),
    )


def lin_at(x, kq, laq):
    """Interpolate log P_lin in ln a at each requested k."""
    out = []
    for kk in kq:
        vals = np.array(
            [np.exp(np.interp(np.log(kk), np.log(x["k"]), np.log(s))) for s in x["s"][1:]]
        )
        out.append(np.exp(np.interp(laq, x["la"][1:], np.log(vals))))
    return np.array(out)


def summarize(name, r):
    print(f"  {name:<26} median {100 * np.median(r):+7.3f}%   "
          f"max|.| {100 * np.max(np.abs(r)):6.3f}%")


def main():
    D = {c: load(c) for c in CASES}
    k = np.exp(np.linspace(np.log(K_WINDOW[0]), np.log(K_WINDOW[1]), 25))
    la = np.linspace(*LA_WINDOW, 6)

    print("=" * 74)
    print(f"zeta_lin, matched (k, ln a): k in {K_WINDOW}, ln a in {LA_WINDOW}")
    print("=" * 74)
    R = {c: lin_at(D[c], k, la) for c in CASES}

    print("\ndt pairs (same N, L, seed):")
    for a, b in [("n128", "dt4_n128"), ("dt4_n128", "dt8_n128")]:
        summarize(f"{b} vs {a}", R[b] / R[a] - 1)

    print("\nresolution (relative to the N=200 dt=1 run):")
    for c in ["n128", "n300", "n400"]:
        r = R[c] / R[REF] - 1
        print(f"  {c:<8} median {100 * np.median(r):+7.2f}%   "
              f"spread(1sigma) {100 * np.std(r):5.2f}%")

    print("\nfinal delta-N spectra, same k window:")
    g = k
    ip = lambda x, xp, fp: np.exp(np.interp(np.log(x), np.log(xp), np.log(fp)))
    P = {c: ip(g, D[c]["k"], D[c]["dn"]) for c in CASES}
    for c in ["n128", "n300", "n400"]:
        print(f"  {c:<8} vs {REF:<10} median {100 * np.median(P[c] / P[REF] - 1):+7.2f}%")
    for a, b in [("n128", "dt4_n128"), ("dt4_n128", "dt8_n128")]:
        print(f"  {b:<8} vs {a:<10} median "
              f"{100 * np.median(P[b] / P[a] - 1):+7.4f}%")

    print("\npeak values (each run's own final snapshot):")
    print(f"  {'case':<10}{'peak P_lin':>12}{'peak P_dN':>12}{'dN/lin @2.4e-8':>17}")
    for c in ["dt1_n200", "n128", "n300", "n400"]:
        x = D[c]
        pl, dn = x["s"][-1], x["dn"]
        j = int(np.argmin(np.abs(x["k"] - 2.4e-8)))
        print(f"  {c:<10}{np.max(pl):>12.4e}{np.max(dn):>12.4e}"
              f"{dn[j] / pl[j]:>17.3f}")

    print("\nzeta_lin is not frozen (n128 peak vs ln a):")
    x = D["n128"]
    for i in range(1, len(x["la"])):
        print(f"   ln a={x['la'][i]:.3f}   peak P_lin={np.max(x['s'][i]):.4e}")

    if "--plot" in sys.argv:
        make_plot(D)


def make_plot(D):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = np.exp(np.linspace(np.log(K_WINDOW[0]), np.log(K_WINDOW[1]), 60))
    la_plot = 9.0
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.2, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )
    ref = lin_at(D[REF], k, la_plot)
    ax1.loglog(k, ref, "k", lw=2.0, label=r"$N=200$, $\zeta_{\rm lin}$")
    for c, col in [("n128", "C3"), ("n300", "C4"), ("n400", "C2")]:
        ax1.loglog(k, lin_at(D[c], k, la_plot), color=col, lw=1.6, label=rf"$N={D[c]['N']}$")
    ax1.loglog(k, lin_at(D["n400"], k, la_plot), "C2--", lw=1.0)
    # dt pair: two nearly identical curves, plotted to show they coincide
    ax1.loglog(k, lin_at(D["dt4_n128"], k, la_plot), "C0:", lw=1.2,
               label=r"$N=128$, $\Delta t=4$")
    ax1.set_ylabel(r"$\mathcal{P}_{\zeta,\rm lin}$")
    ax1.legend(frameon=False, fontsize=8, ncol=2, loc="lower left",
               bbox_to_anchor=(0.0, 1.01))
    ax1.tick_params(labelbottom=False)
    style_axes(ax1)
    style_log_xaxis(ax1)

    ax2.axhline(0, color="k", lw=0.8)
    for c, col in [("n128", "C3"), ("n300", "C4"), ("n400", "C2")]:
        ax2.semilogx(k, 100 * (lin_at(D[c], k, la_plot) / ref - 1), color=col, lw=1.5,
                     label=rf"$N={D[c]['N']}$")
    ax2.semilogx(k, 100 * (lin_at(D["dt4_n128"], k, la_plot) /
                           lin_at(D["n128"], k, la_plot) - 1), "C0:", lw=1.4,
                 label=r"$N=128$: $\Delta t=4$ vs $\Delta t=2$")
    ax2.set_ylabel(r"$\mathcal{P}/\mathcal{P}_{N=200}-1\ [\%]$")
    ax2.set_xlabel(r"$k\ [\mathrm{Hz}]$")
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    style_axes(ax2)
    style_log_xaxis(ax2)

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"zeta-lin.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote", out)


if __name__ == "__main__":
    main()
