"""Observed vs modelled n_s(T), both defined the same way:

    n_s = N(> 20 um) / S_dust(t)

with S_dust(t) = KIT's initial dust surface (InitialPNSD split) x the observed
CPC dilution ratio, for observations AND model, so that differences are purely
differences in the number of particles > 20 um (model: ice + liquid > 20 um, as
KIT's NIce counts every particle > 20 um).

Also prints the model's own initial dust surface (from the lognormal modes in the
generated namelist) against KIT's split, as a check.

Usage: python3 ns_compare.py [RUNS_DIR]
Outputs: figs/ns_compare.png, figs/ns_compare_ratio.png, figs/ns_compare.csv
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ns_observed as obsmod   # noqa: E402
import compare_mp as cmp       # noqa: E402

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "iSKYLAB01/runs"
cmp.RUNS = RUNS
OUT = HERE / "figs"
MODELS = [("mp_base", "model, wetting mode 0", "-"), ("mp_wet1", "model, wetting mode 1", "--")]
DUSTLESS = {"mp_base": "mp_dust", "mp_wet1": "mp_wet1dust"}


def model_dust_surface(tag, exp):
    """Initial dust surface (um2 cm-3) from the dust mode in the namelist."""
    f = RUNS / tag / "single_comparison" / f"namelist-{exp}.in"
    txt = f.read_text()
    def arr(name):
        m = re.search(rf"{name}\(1:3,1:1\)\s*=\s*([^,]+),\s*([^,]+),\s*([^,\n]+)", txt)
        return np.array([float(m.group(i)) for i in (1, 2, 3)])
    n, d, sig = arr("n_aer1"), arr("d_aer1"), arr("sig_aer1")
    p = float(re.search(r"pinit\s*=\s*([0-9.eE+-]+)", txt).group(1))
    t = float(re.search(r"tinit\s*=\s*([0-9.eE+-]+)", txt).group(1))
    rhoa = p / (287.05 * t)
    s = np.sum(n * rhoa * np.pi * d ** 2 * np.exp(2 * sig ** 2))   # m2 m-3
    return s * 1e6  # um2 cm-3


def ns_series(m, d_obs, s0):
    """Model N(>20 um) on the observation times divided by the observed S_dust(t)."""
    ni = np.interp(d_obs["t"], m["t"], m["ni"], right=np.nan)
    return ni * 1e6 / (s0 * 1e-12 * 1e6 * d_obs["f_cpc"])


def main():
    tmid = 0.5 * (obsmod.TBINS[1:] + obsmod.TBINS[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    figr, axr = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    rows = []
    print("Initial dust surface (um2 cm-3): KIT split vs model namelist (mp_base)")
    for ax, axq, (dust, runs) in zip(axes, axr, obsmod.RUNS.items()):
        refs = {}
        for (exp, label, nacl), col in zip(runs, obsmod.COLORS):
            s0, _ = obsmod.initial_surface(exp)
            d = obsmod.series(exp)
            cloud = d["t"] > 0
            try:
                sm = model_dust_surface("mp_base", exp)
            except Exception:
                sm = np.nan
            print(f"  {exp}: KIT {s0:6.0f}  model {sm:6.0f}  (model/KIT {sm/s0:.2f})")
            ns_o = d["ni"] * 1e6 / (s0 * 1e-12 * 1e6 * d["f_cpc"])
            med_o = obsmod.binned(d["tc"], ns_o, cloud)
            ax.semilogy(tmid, med_o, "-", color=col, lw=2.5, alpha=0.35)
            series = {"obs": med_o}
            for tag, lab, ls in MODELS:
                m = cmp.model(tag, exp)
                if m is None:
                    continue
                med_m = obsmod.binned(d["tc"], ns_series(m, d, s0), cloud)
                series[tag] = med_m
                ax.semilogy(tmid, med_m, ls, color=col, lw=1.4)
            if nacl == 0:
                refs = series
            else:
                for key, ls, lw, a in (("obs", "-", 2.5, 0.35), ("mp_base", "-", 1.4, 1), ("mp_wet1", "--", 1.4, 1)):
                    if key in series and key in refs:
                        axq.plot(tmid, series[key] / refs[key], ls, color=col, lw=lw, alpha=a)
            for i, tm in enumerate(tmid):
                rows.append([dust, exp, f"{tm:.2f}"] + [f"{series.get(k, np.full(len(tmid), np.nan))[i]:.3e}" for k in ("obs", "mp_base", "mp_wet1")])
        ax.set_title(f"{dust}: n$_s$ = N(>20 µm) / S$_{{dust}}$")
        for a in (ax, axq):
            a.set_xlim(-30.5, -16); a.invert_xaxis(); a.grid(alpha=0.3, which="both")
            a.set_xlabel("Gas temperature (°C)")
        axq.axhline(1, color="0.5", lw=1); axq.axhline(0.5, color="0.5", lw=1, ls=":")
        axq.set_yscale("log"); axq.set_ylim(1e-3, 30)
        axq.set_title(f"{dust}: + NaCl / dust alone")
    # legends
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="0.3", lw=2.5, alpha=0.35, label="observed (KIT WELAS)"),
         Line2D([], [], color="0.3", lw=1.4, ls="-", label="model, wetting mode 0 (dust must activate)"),
         Line2D([], [], color="0.3", lw=1.4, ls="--", label="model, wetting mode 1 (X ≥ 0.01)")]
    for (dust, runs), ax in zip(obsmod.RUNS.items(), axes):
        hh = h + [Line2D([], [], color=c, lw=3, label=f"{e} {l}") for (e, l, _), c in zip(runs, obsmod.COLORS)]
        ax.legend(handles=hh, fontsize=7, loc="lower right")
    axr[0].legend(handles=h, fontsize=7, loc="lower left")
    axes[0].set_ylabel("n$_s$ (m⁻²)"); axr[0].set_ylabel("ratio")
    for f_, name in ((fig, "ns_compare.png"), (figr, "ns_compare_ratio.png")):
        f_.tight_layout(); f_.savefig(OUT / name, dpi=150)
    with open(OUT / "ns_compare.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["dust", "exp", "T_mid_C", "ns_obs", "ns_mode0", "ns_mode1"]); w.writerows(rows)
    print("written", OUT)


if __name__ == "__main__":
    main()
