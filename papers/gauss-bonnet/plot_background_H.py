#!/usr/bin/env python3
"""Compare the background expansion rate H(t) for different lattice sizes.

H is reconstructed from the stored measurables as H = v_a / a (code units).
N=300 and N=400 have full time series in their ``summary.npz``.  The N=200
production run only saved spectra, so its curve is *inferred* from the N=300
and N=400 runs assuming that the initial offset scales as N^4:

    H_200(t) = H_300(t) + r [H_400(t) - H_300(t)],
    r = (200^4 - 300^4) / (400^4 - 300^4).

The N^4 law is checked with N=64 and N=128, which do have time series.
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
from matplotlib.ticker import PercentFormatter

DATA = ROOT / "inflat/out/gauss_bonnet_convergence/alpha_1.13412e7"
FIG_DIR = Path(__file__).resolve().parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CASES = ["n64", "n128", "n300", "n400"]
COLORS = {"n64": "0.65", "n128": "0.45", "n200": "#17becf",
          "n300": "#9467bd", "n400": "#2ca02c"}


def load(name):
    d = np.load(DATA / f"{name}.summary.npz")
    a = d["measurable_a"]
    v_a = d["measurable_v_a"]
    return {"n": int(d["N"][0]), "ln_a": np.log(a), "H": v_a / a}


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.8,
        }
    )

    data = {c: load(c) for c in CASES}
    n_of = {c: data[c]["n"] for c in CASES}

    # Background H(t) from the same calculate_background used by the linear
    # (perturbation) calculation.  This is the reference curve.
    bg = np.load(DATA / "background.npz")
    bg_ln_a = bg["ln_a"]
    bg_H = bg["H"]
    h_bg = float(np.interp(0.0, bg_ln_a, bg_H))

    # Initial H and the N^4 amplitude (fit through H_bg).
    ns = np.array([n_of[c] for c in CASES], float)
    h0 = np.array([data[c]["H"][0] for c in CASES])
    x4 = ns**4
    slope = float(np.sum(x4 * (h0 - h_bg)) / np.sum(x4 * x4))

    print(f"background H(0) = {h_bg:.6e},  A = {slope:.4e}")
    for n, h in zip(ns, h0):
        print(f"  N={int(n):3d}  dH_ini/H_bg = {100 * (h / h_bg - 1):.5f}%")

    n_inf = 200
    print(f"  N=200 (inferred) dH_ini/H_bg = "
          f"{100 * (slope * n_inf**4 / h_bg):.5f}%")

    # Common time grid.
    lo = max(d["ln_a"][0] for d in data.values())
    hi = min(d["ln_a"][-1] for d in data.values())
    grid = np.linspace(lo, hi, 4000)
    Hg = {c: np.interp(grid, data[c]["ln_a"], data[c]["H"]) for c in CASES}
    bg_grid = np.interp(grid, bg_ln_a, bg_H)

    # Inferred N=200 curve.
    r = (n_inf**4 - 300**4) / (400**4 - 300**4)
    H200 = Hg["n300"] + r * (Hg["n400"] - Hg["n300"])

    # Validate the same interpolation at N=128.
    r128 = (128**4 - 300**4) / (400**4 - 300**4)
    H128_pred = Hg["n300"] + r128 * (Hg["n400"] - Hg["n300"])
    err128 = np.max(np.abs(H128_pred / Hg["n128"] - 1))
    print(f"interpolation check at N=128: max |pred/meas - 1| = {err128:.2e}")

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(11.5, 3.8),
        gridspec_kw={"width_ratios": [1.25, 1.25, 0.95], "wspace": 0.42},
    )

    # ---- panel 1: absolute H(t) ---------------------------------------
    ax1.plot(grid, bg_grid, color="k", lw=1.3,
             label="linear background")
    ax1.plot(grid, Hg["n300"], color=COLORS["n300"], label=r"$N=300$")
    ax1.plot(grid, Hg["n400"], color=COLORS["n400"], label=r"$N=400$")
    ax1.plot(grid, H200, color=COLORS["n200"], ls="--",
             label=r"$N=200$ (inferred)")
    ax1.set_xlabel(r"$\ln a$")
    ax1.set_ylabel(r"$H$ (code units)")
    ax1.set_xlim(grid[0], grid[-1])
    ax1.legend(frameon=False, fontsize=7.5)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))
    ax1.yaxis.get_offset_text().set_fontsize(8)

    # ---- panel 2: offsets relative to the linear background -----------
    ax2.axhline(0.0, color="k", ls=":", lw=1.0)
    ax2.plot(grid, Hg["n400"] / bg_grid - 1,
             color=COLORS["n400"], label=r"$N=400$")
    ax2.plot(grid, H200 / bg_grid - 1, color=COLORS["n200"],
             ls="--", label=r"$N=200$ (inferred)")
    ax2.plot(grid, Hg["n128"] / bg_grid - 1,
             color=COLORS["n128"], lw=0.9, ls="-",
             label=r"$N=128$ (measured)")
    ax2.plot(grid, Hg["n64"] / bg_grid - 1,
             color=COLORS["n64"], lw=0.9, ls="-",
             label=r"$N=64$ (measured)")
    ax2.set_xlim(0.0, 3.0)
    ax2.set_xlabel(r"$\ln a$")
    ax2.set_ylabel(r"$H_N/H_{\rm bg}-1$")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax2.legend(frameon=False, fontsize=7.5, loc="upper right")

    # ---- panel 3: initial offset vs N ---------------------------------
    nn = np.array([64, 128, 200, 300, 400], float)
    off = np.array([slope * x**4 / h_bg for x in nn])
    ax3.loglog(nn, off, "o", color="0.25", ms=4)
    ax3.loglog([nn[2]], [off[2]], "o", mfc="white", mec=COLORS["n200"],
               ms=7, mew=1.5)
    nn_line = np.logspace(np.log10(50), np.log10(500), 50)
    ax3.loglog(nn_line, slope * nn_line**4 / h_bg, "--",
               color="0.55", lw=1.2, label=r"$\propto N^4$")
    ax3.set_xlabel(r"$N$")
    ax3.set_ylabel(r"$\delta H_{\rm ini}/H_{\rm bg}$")
    ax3.set_xticks([64, 128, 200, 300, 400])
    ax3.set_xticklabels(["64", "128", "200", "300", "400"], rotation=45,
                        ha="right", fontsize=8)
    ax3.minorticks_off()
    ax3.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    ax3.legend(frameon=False, fontsize=8)
    ax3.text(60, off[-1] * 0.45, r"$\delta H/H_{\rm bg}\propto N^4$",
             fontsize=7.5, color="0.4")

    fig.suptitle(
        r"background $H(t)$, $\alpha=1.13412\times10^{7}$ (all $N$ share one box)",
        fontsize=10, y=1.02,
    )
    fig.savefig(FIG_DIR / "background-H-n200-n300-n400.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "background-H-n200-n300-n400.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {FIG_DIR / 'background-H-n200-n300-n400.pdf'}")


if __name__ == "__main__":
    main()
