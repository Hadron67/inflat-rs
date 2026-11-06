#!/usr/bin/env python3
"""Background evolution comparison for the initial-H diagnostic.

Produces:

  * background-h:    H(ln a) and H/H_bg - 1 for N=128, N=300, N=400 and the
                     `n128_h400` run (N=128 started from the N=400 H_ini).
  * background-vphi: <phi-dot>(ln a) and its ratio to N=128, which explains
                     why `n128_h400` differs so strongly in the spectrum:
                     the higher initial H increases the Hubble friction and
                     keeps the field in the USR phase much longer.

The linear background comes from `background.npz` (written by
`gauss_bonnet_convergence --dump-background`).
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
from matplotlib.ticker import (
    LogLocator,
    NullFormatter,
    PercentFormatter,
    ScalarFormatter,
)

DATA = ROOT / "inflat/out/gauss_bonnet_convergence/alpha_1.13412e7"
FIG_DIR = Path(__file__).resolve().parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    ("n128", r"$N=128$", "C3", "-"),
    ("n300", r"$N=300$", "C4", "-"),
    ("n400", r"$N=400$", "C2", "-"),
    ("n128_h400", r"$N=128,\ H_{\rm ini}=H_{\rm ini}(N=400)$", "C1", "--"),
]


def load_case(name):
    path = DATA / f"{name}.summary.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    data = np.load(path)
    a = data["measurable_a"]
    return {
        "ln_a": np.log(a),
        "H": data["measurable_v_a"] / a,
        "v_phi": data["measurable_v_phi"],
        "phi": data["measurable_phi"],
    }


def load_linear_background():
    path = DATA / "background.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    data = np.load(path)
    return data["ln_a"], data["H"], data["v_phi"]


def style_axes(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def style_log_xaxis(ax):
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20)
    )
    ax.xaxis.set_minor_formatter(NullFormatter())


def setup_rc():
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


def make_h_plot(cases, ln_bg, H_bg):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(6.2, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )

    ax1.plot(ln_bg, H_bg, color="k", ls="--", lw=1.3, label="linear background")
    for name, label, color, ls in CASES:
        ax1.plot(cases[name]["ln_a"], cases[name]["H"], color=color, ls=ls, lw=1.6, label=label)

    ax1.set_ylabel(r"$H$")
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))
    ax1.set_ylim(2.62e-5, 2.73e-5)
    ax1.legend(loc="lower left", frameon=False, ncol=1)
    ax1.tick_params(labelbottom=False)
    style_axes(ax1)

    for name, label, color, ls in CASES:
        ln = cases[name]["ln_a"]
        rel = cases[name]["H"] / np.interp(ln, ln_bg, H_bg) - 1.0
        ax2.plot(ln, rel, color=color, ls=ls, lw=1.5, label=label)

    ax2.axhline(0.0, color="k", ls=":", lw=1.0)
    ax2.set_xlabel(r"$\ln a$")
    ax2.set_ylabel(r"$H/H_{\rm bg}-1$")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    ax2.set_ylim(-1.0e-3, 7.5e-3)
    style_axes(ax2)

    ax1.set_xlim(0.0, 11.0)
    fig.align_ylabels([ax1, ax2])
    fig.savefig(FIG_DIR / "background-h.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "background-h.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_vphi_plot(cases, ln_bg, vphi_bg):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(6.2, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )

    ax1.plot(ln_bg, np.abs(vphi_bg), color="k", ls="--", lw=1.3, label="linear background")
    for name, label, color, ls in CASES:
        ax1.plot(
            cases[name]["ln_a"],
            np.abs(cases[name]["v_phi"]),
            color=color,
            ls=ls,
            lw=1.6,
            label=label,
        )

    ax1.set_yscale("log")
    ax1.set_ylabel(r"$|\dot\phi|$")
    ax1.set_ylim(5e-10, 6e-6)
    ax1.legend(loc="lower left", frameon=False, ncol=1)
    ax1.tick_params(labelbottom=False)
    style_axes(ax1)

    ref = cases["n128"]
    for name, label, color, ls in CASES:
        if name == "n128":
            continue
        ln = cases[name]["ln_a"]
        mask = ln <= ref["ln_a"][-1]
        ratio = np.abs(cases[name]["v_phi"][mask]) / np.abs(
            np.interp(ln[mask], ref["ln_a"], ref["v_phi"])
        )
        ax2.semilogy(ln[mask], ratio, color=color, ls=ls, lw=1.5, label=label)

    ax2.axhline(1.0, color="k", ls=":", lw=1.0)
    ax2.set_xlabel(r"$\ln a$")
    ax2.set_ylabel(r"$|\dot\phi|/|\dot\phi|_{N=128}$")
    style_axes(ax2)

    ax1.set_xlim(0.0, 11.0)
    fig.align_ylabels([ax1, ax2])
    fig.savefig(FIG_DIR / "background-vphi.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "background-vphi.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    setup_rc()
    cases = {name: load_case(name) for name, _, _, _ in CASES}
    ln_bg, H_bg, vphi_bg = load_linear_background()
    bg_mask = ln_bg <= 11.0
    ln_bg, H_bg, vphi_bg = ln_bg[bg_mask], H_bg[bg_mask], vphi_bg[bg_mask]

    make_h_plot(cases, ln_bg, H_bg)
    make_vphi_plot(cases, ln_bg, vphi_bg)

    print("H/H_bg - 1 at selected ln a:")
    header = "ln a  " + "".join(f"{name:>12}" for name, _, _, _ in CASES)
    print(header)
    for target in [0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0]:
        row = f"{target:5.1f}  "
        for name, _, _, _ in CASES:
            ln = cases[name]["ln_a"]
            if target > ln[-1]:
                row += f"{'--':>12}"
                continue
            h = np.interp(target, ln, cases[name]["H"])
            h_bg = np.interp(target, ln_bg, H_bg)
            row += f"{h / h_bg - 1.0:>12.3%}"
        print(row)

    print("\n|phi-dot| / |phi-dot|_N128 at selected ln a:")
    print(header)
    ref = cases["n128"]
    for target in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.8]:
        row = f"{target:5.1f}  "
        for name, _, _, _ in CASES:
            ln = cases[name]["ln_a"]
            if target > ln[-1] or target > ref["ln_a"][-1]:
                row += f"{'--':>12}"
                continue
            v = np.interp(target, ln, np.abs(cases[name]["v_phi"]))
            v_ref = np.interp(target, ref["ln_a"], np.abs(ref["v_phi"]))
            row += f"{v / v_ref:>12.3f}"
        print(row)

    print(f"\nwrote {FIG_DIR / 'background-h.pdf'}")
    print(f"wrote {FIG_DIR / 'background-vphi.pdf'}")


if __name__ == "__main__":
    main()
