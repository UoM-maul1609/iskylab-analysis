"""Observed ice-nucleation active-site density n_s(T) for the iSKYLAB mixed-phase
runs, computed from the KIT data files ourselves.

    n_s(t) = NIce(t) / S_dust(t)

NIce     : KIT merged WELAS, particles > 20 um (counts large droplets as ice while
           liquid is present; periods with LWC > 0 are marked).
S_dust(0): dust surface area at expansion start from the InitialPNSD-Mrg files
           (in-chamber #/cc rows).  For the mixtures, KIT's dust component of the
           merged PSD; the total (dust + NaCl) is also written for comparison.
S_dust(t): S_dust(0) x f(t), with two choices of f:
           'cpc'      : CPC_TotBot(t)/CPC_TotBot(0) - dilution plus all particle
                        losses (main choice);
           'pressure' : (p/T)(t)/(p/T)(0) - pure expansion dilution (no losses;
                        gives the lowest n_s).
NIce is airborne ice only: crystals that have already fallen out are not counted,
so n_s from airborne ice is a lower bound on the cumulative n_s.

Outputs (analysis/mixed_phase/figs/):
  ns_observed.png      n_s(T) per dust type, dust alone vs + NaCl
  ns_ratio.png         n_s(mixture)/n_s(dust alone) vs T
  ns_observed.csv      0.5 K binned medians for every run and choice

Usage: python3 ns_observed.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "iSKYLAB-data/Datasets-V2"
TS = DATA / "Timeseries-During-Expansion"
PNSD = DATA / "Initial-PNSD"
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)

RUNS = {
    "ATD03": [("Exp025", "ATD alone", 0), ("Exp026", "+ NaCl (4000 req.)", 4000), ("Exp027", "+ NaCl (10000 req.)", 10000)],
    "SDSA01": [("Exp021", "SDSA alone", 0), ("Exp022", "+ NaCl (2000 req.)", 2000), ("Exp023", "+ NaCl (2000 req.)", 2000)],
}
DUST_FIT = {  # geometric-area n_s used in BMM, m-2 (Tc in deg C)
    "ATD03": (lambda tc: np.exp(-0.380 * tc + 13.918), (-26.7, -17.7), "Niemand et al. (2012) ATD"),
    "SDSA01": (lambda tc: np.exp(-0.517 * tc + 8.934), (-36.0, -12.0), "Niemand et al. (2012) desert dust"),
}
COLORS = ["#1b1b1b", "#d1495b", "#2e86ab"]
TBINS = np.arange(-30.5, -15.0, 0.5)


def _num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except ValueError:
        return np.nan


def initial_surface(exp):
    """Return (dust S, total S) in um2 cm-3 at expansion start (in-chamber rows)."""
    rows = [r for r in csv.reader(open(PNSD / f"iSKYLAB01-{exp}-3-InitialPNSD-Mrg.csv")) if r]
    d = np.array([_num(v) for v in rows[0][1:]])
    dlog = np.nan_to_num(np.array([_num(v) for v in rows[1][1:]]))
    comp = {}
    for r in rows[2:]:
        if "#/cc" not in r[0]:
            continue
        n = np.nan_to_num(np.array([_num(v) for v in r[1:]]))
        k = min(len(n), len(d), len(dlog))
        comp[r[0].split()[0]] = float(np.sum(n[:k] * np.pi * d[:k] ** 2 * dlog[:k]))
    total = comp["Total"]
    parts = [v for k, v in comp.items() if k not in ("Total", "NaCl")]
    dust = sum(parts) if parts else total   # single-component runs: Total is dust
    return dust, total


def series(exp):
    m = list(csv.reader(open(TS / f"iSKYLAB01-{exp}-2-OPC-d-MergedW.csv")))
    h = [x.strip() for x in m[0]]
    col = lambda p: [i for i, x in enumerate(h) if x.startswith(p)][0]
    w = np.array([[_num(r[0]), _num(r[col("NIce")]), _num(r[col("LWC")])] for r in m[1:]])
    c = list(csv.reader(open(TS / f"iSKYLAB01-{exp}-1-MeteoCPC.csv")))
    hc = [x.strip() for x in c[0]]
    cc = lambda p: [i for i, x in enumerate(hc) if x.startswith(p)][0]
    met = np.array([[_num(r[0]), _num(r[cc("Pressure")]), _num(r[cc("Tgw_mean")]), _num(r[cc("CPC_TotBot")])] for r in c[1:]])
    t = w[:, 0]
    p = np.interp(t, met[:, 0], met[:, 1])
    tc = np.interp(t, met[:, 0], met[:, 2])
    good = np.isfinite(met[:, 3])
    cpc = np.interp(t, met[good, 0], met[good, 3])
    k = 61  # 61 s running mean of the CPC ratio
    cpc = np.convolve(np.pad(cpc, k // 2, mode="edge"), np.ones(k) / k, "valid")
    i0 = np.argmin(np.abs(t))
    f_cpc = cpc / cpc[i0]
    f_p = (p / (tc + 273.15)) / (p[i0] / (tc[i0] + 273.15))
    ni = w[:, 1]
    ni_s = np.convolve(np.pad(np.nan_to_num(ni), 15, mode="edge"), np.ones(31) / 31, "valid")
    ni_s[~np.isfinite(ni)] = np.nan
    return {"t": t, "tc": tc, "ni": ni_s, "lwc": w[:, 2], "f_cpc": f_cpc, "f_pressure": f_p}


def binned(tc, y, mask):
    med = np.full(len(TBINS) - 1, np.nan)
    for i in range(len(TBINS) - 1):
        s = mask & (tc >= TBINS[i]) & (tc < TBINS[i + 1]) & np.isfinite(y) & (y > 0)
        if s.sum() >= 10:
            med[i] = np.median(y[s])
    return med


def main():
    tmid = 0.5 * (TBINS[1:] + TBINS[:-1])
    table = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    figr, axr = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, axq, (dust, runs) in zip(axes, axr, RUNS.items()):
        ref = None
        for (exp, label, nacl), col in zip(runs, COLORS):
            s_dust, s_tot = initial_surface(exp)
            d = series(exp)
            cloud = d["t"] > 0
            liquid = np.nan_to_num(d["lwc"]) > 0.001
            for choice in ("cpc", "pressure"):
                ns = d["ni"] * 1e6 / (s_dust * 1e-12 * 1e6 * d[f"f_{choice}"])  # m-2
                med = binned(d["tc"], ns, cloud)
                med_ice = binned(d["tc"], ns, cloud & ~liquid)
                ns_tot = d["ni"] * 1e6 / (s_tot * 1e-12 * 1e6 * d[f"f_{choice}"])
                med_tot = binned(d["tc"], ns_tot, cloud)
                for i, tm in enumerate(tmid):
                    table.append([dust, exp, choice, f"{tm:.2f}", f"{s_dust:.1f}", f"{s_tot:.1f}",
                                  f"{med[i]:.3e}", f"{med_ice[i]:.3e}", f"{med_tot[i]:.3e}"])
                if choice == "cpc":
                    ax.semilogy(tmid, med, "-", color=col, lw=2, label=f"{exp} {label}  (S$_{{dust}}$={s_dust:.0f} µm² cm⁻³)")
                    ax.semilogy(tmid, med_ice, "o", color=col, ms=4)
                    if nacl == 0:
                        ref = med
                    else:
                        axq.plot(tmid, med / ref, "-", color=col, lw=2, label=f"{exp} {label}")
                else:
                    ax.semilogy(tmid, med, ":", color=col, lw=1)
        fn, (lo, hi), name = DUST_FIT[dust]
        tt = np.linspace(max(lo, -30), min(hi, -15), 50)
        ax.semilogy(tt, fn(tt), "--", color="0.6", lw=1, label=f"{name} (BMM)")
        ax.set_title(f"{dust}: n$_s$ from airborne ice, divided by dust surface")
        ax.set_xlabel("Gas temperature (°C)")
        ax.set_xlim(-30.5, -16)
        ax.invert_xaxis()
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower right")
        axq.axhline(1, color="0.5", lw=1)
        axq.axhline(0.5, color="0.5", lw=1, ls="--")
        axq.set_yscale("log")
        axq.set_ylim(0.05, 20)
        axq.set_xlim(-30.5, -16)
        axq.invert_xaxis()
        axq.set_xlabel("Gas temperature (°C)")
        axq.set_title(f"{dust}: n$_s$(+ NaCl) / n$_s$(dust alone)")
        axq.grid(alpha=0.3, which="both")
        axq.legend(fontsize=7)
    axes[0].set_ylabel("n$_s$ (m⁻²)")
    axr[0].set_ylabel("ratio")
    fig.text(0.01, 0.005, "Lines: 0.5 K medians, dust surface diluted with the CPC ratio. Dotted: pure pressure dilution. "
             "Dots: only times with LWC ≤ 0.001 g m⁻³ (NIce cannot include droplets).", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    figr.tight_layout()
    fig.savefig(OUT / "ns_observed.png", dpi=150)
    figr.savefig(OUT / "ns_ratio.png", dpi=150)
    with open(OUT / "ns_observed.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dust", "exp", "dilution", "T_mid_C", "S_dust0_um2cm3", "S_total0_um2cm3",
                    "ns_dust_m2", "ns_dust_noliquid_m2", "ns_total_m2"])
        w.writerows(table)
    print("written", OUT)


if __name__ == "__main__":
    main()
