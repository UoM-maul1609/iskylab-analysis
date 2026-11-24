"""Figures and numbers for the iSKYLAB mixed-phase paper (draft v0.1, 16 Nov 2026).

Reads the KIT Datasets-V2 files and the BMM runs of the `final` set
(scripts/batchRuns_v2.sh set_final) and writes
    analysis/mixed_phase/paper/figs/*.pdf|png
    analysis/mixed_phase/paper/numbers.json   (every number quoted in the draft)

Definitions (kept identical for observations and model):
  * WELAS "droplets": particles 0.39-13 um (KIT NDrop); model diameters are
    passed through the inlet operator D_obs^2 = D^2 - DD2 (DD2 = 25 um^2 for the
    cold runs; physical estimate for the room-temperature WELAS).
  * "Ice": particles > 20 um (KIT NIce uses > 13 um; the WELAS spectra have a
    gap between 12 and 25 um; the model threshold sensitivity is reported).
  * Ice window: t_sat + 300 s to the end of the expansion.  Droplet window:
    t_sat to t_sat + 600 s.
  * Cross-experiment ratio R = <Ni>(mixture) / <Ni>(dust alone), computed the
    same way for observations and model (different dust loads in each run).
  * Paired ratio P = <Ni_all>(mixture) / <Ni_all>(same run, NaCl removed):
    model only.

Usage: python3 paper_figs.py [RUNS_DIR]
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "iSKYLAB-data/Datasets-V2"
TS = DATA / "Timeseries-During-Expansion"
PNSD = DATA / "Initial-PNSD"
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "iSKYLAB01/runs"
OUT = HERE / "paper"
FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)

EXPS = ["Exp021", "Exp022", "Exp023", "Exp025", "Exp026", "Exp027"]
MIX = ["Exp022", "Exp023", "Exp026", "Exp027"]
ALONE = {"Exp022": "Exp021", "Exp023": "Exp021", "Exp026": "Exp025", "Exp027": "Exp025"}
DUST = {"Exp021": "SDSA01", "Exp022": "SDSA01", "Exp023": "SDSA01",
        "Exp025": "ATD03", "Exp026": "ATD03", "Exp027": "ATD03"}
TSAT = {"Exp021": 720, "Exp022": 840, "Exp023": 780, "Exp025": 600, "Exp026": 720, "Exp027": 840}
# ISKYLAB_TSAT=hygrometer: windows start when the hygrometer total water reaches water
# saturation (measured), not at the listed visible-cloud onset (16 Nov 2026).
TSAT_HYGROMETER = {"Exp021": 672, "Exp022": 738, "Exp023": 570, "Exp025": 546, "Exp026": 492, "Exp027": 552}
import os as _os
if _os.environ.get("ISKYLAB_TSAT", "").strip() == "hygrometer":
    TSAT = dict(TSAT_HYGROMETER)
DTBL = {"Exp021": 0.13, "Exp022": 0.0, "Exp023": 0.0, "Exp025": 0.0, "Exp026": 0.0, "Exp027": 0.0}
DD2 = 25e-12
DMIN, DDROP, DICE = 0.39e-6, 13e-6, 20e-6

INK, MUTED = "#1f1f1f", "#8a8a85"
C_RUN = {"Exp021": INK, "Exp025": INK, "Exp022": "#2a78d6", "Exp026": "#2a78d6",
         "Exp023": "#eb6834", "Exp027": "#eb6834"}
C_M0, C_M1, C_X1 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 200})


def _num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except ValueError:
        return np.nan


def ew(T):
    return np.exp(54.842763 - 6763.22 / T - 4.210 * np.log(T) + 0.000367 * T
                  + np.tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T))


def ei(T):
    return np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T)


def smooth(y, k):
    y = np.asarray(y, float)
    m = np.isfinite(y)
    num = np.convolve(np.where(m, y, 0.0), np.ones(k), "same")
    den = np.convolve(m.astype(float), np.ones(k), "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ----------------------------------------------------------------- observations
def obs(exp):
    c = list(csv.reader(open(TS / f"iSKYLAB01-{exp}-1-MeteoCPC.csv")))
    hc = [x.strip() for x in c[0]]
    ci = lambda p: [i for i, x in enumerate(hc) if x.startswith(p)][0]
    met = np.array([[_num(r[i]) for i in (0, ci("Pressure"), ci("Tgw_mean"), ci("Tww_mean"),
                                           ci("TDew"), ci("TFrost"), ci("CPC_TotBot"))] for r in c[1:]])
    w = list(csv.reader(open(TS / f"iSKYLAB01-{exp}-2-OPC-d-MergedW.csv")))
    hw = [x.strip() for x in w[0]]
    nbin = [i for i, x in enumerate(hw) if x.startswith("NTot")][0] - 1
    D = np.array([_num(x) for x in hw[1:1 + nbin]])
    arr = np.array([[_num(v) for v in r[:1 + nbin + 4]] for r in w[1:] if len(r) >= nbin + 5])
    t = arr[:, 0]
    o = {"t": t, "D": D, "psd": arr[:, 1:1 + nbin], "nd": arr[:, 1 + nbin + 1],
         "ni": arr[:, 1 + nbin + 2], "lwc": arr[:, 1 + nbin + 3]}
    for k, j in (("p", 1), ("tg", 2), ("tw", 3), ("tdew", 4), ("tfrost", 5)):
        o[k] = np.interp(t, met[:, 0], met[:, j])
    good = np.isfinite(met[:, 6])
    o["cpc"] = smooth(np.interp(t, met[good, 0], met[good, 6]), 61)
    T = o["tg"] + 273.15
    o["rhw"] = 100 * ew(o["tdew"] + 273.15) / ew(T)       # MBW total water (vapour + condensate)
    o["rhi"] = 100 * ei(o["tfrost"] + 273.15) / ei(T)
    o["pre"] = (t >= -60) & (t <= 0)
    if not np.any(np.isfinite(o["nd"][o["pre"]])):      # Exp027: WELAS starts at t = 0
        o["pre"] = (t >= 0) & (t <= 60)
    o["tend"] = float(np.nanmax(t))
    return o


def initial_psd(exp):
    rows = [r for r in csv.reader(open(PNSD / f"iSKYLAB01-{exp}-3-InitialPNSD-Mrg.csv")) if r]
    d = np.array([_num(v) for v in rows[0][1:]])
    dlog = np.nan_to_num(np.array([_num(v) for v in rows[1][1:]]))
    comp = {}
    for r in rows[2:]:
        if "#/cc" in r[0]:
            comp[r[0].split()[0]] = np.nan_to_num(np.array([_num(v) for v in r[1:]]))
    k = min(len(d), len(dlog), *[len(v) for v in comp.values()])
    d, dlog = d[:k], dlog[:k]
    comp = {key: v[:k] for key, v in comp.items()}
    dust = sum(v for key, v in comp.items() if key not in ("Total", "NaCl")) if len(comp) > 1 else comp["Total"]
    nacl = comp.get("NaCl", np.zeros(k))
    S = lambda n: float(np.sum(n * np.pi * d ** 2 * dlog))
    N = lambda n, lo=0.0: float(np.sum(n * dlog * (d >= lo)))
    return {"d": d, "dust": dust, "nacl": nacl, "S_dust": S(dust), "S_nacl": S(nacl),
            "N_dust": N(dust), "N_nacl": N(nacl), "N_dust_05": N(dust, 0.5), "N_nacl_05": N(nacl, 0.5)}


# ----------------------------------------------------------------------- model
_cache = {}


def model(tag, exp):
    key = (tag, exp)
    if key in _cache:
        return _cache[key]
    f = RUNS / tag / "single_comparison" / f"output-{exp}.nc"
    if not f.exists():
        _cache[key] = None
        return None
    v = Dataset(f).variables
    rhoa = np.asarray(v["p"][:] / (287.05 * v["t"][:]))
    dw, nw = np.asarray(v["dwet"][:]), np.asarray(v["nwat"][:])
    di, ni = np.asarray(v["dmaxice"][:]), np.asarray(v["nicem"][:])
    dobs = np.sqrt(np.clip(dw ** 2 - DD2, 0.0, None))
    ax = (1, 2)
    m = {"t": np.asarray(v["time"][:]), "tg": np.asarray(v["t"][:]) - 273.15, "rh": np.asarray(v["rh"][:]),
         "rhoa": rhoa, "lwc": np.asarray(v["ql"][:]) * rhoa * 1e3,
         "nd": np.sum(nw * ((dobs > DMIN) & (dobs <= DDROP)), axis=ax) * rhoa / 1e6,
         "ni_all": np.asarray(v["nice"][:]) * rhoa / 1e6}
    for thr in (13e-6, 20e-6, 25e-6):
        big = np.sum(nw * (dw > thr) * (dw < 1e-2), axis=ax) + np.sum(ni * (di > thr) * (di < 1e-2), axis=ax)
        m[f"nbig{int(thr * 1e6)}"] = big * rhoa / 1e6           # liquid + ice above threshold
        m[f"ni{int(thr * 1e6)}"] = np.sum(ni * (di > thr) * (di < 1e-2), axis=ax) * rhoa / 1e6   # ice only
    m["ni"] = m["ni20"]
    # WELAS-equivalent liquid water: inlet operator applied, 0.39-13 um
    win = (dobs > DMIN) & (dobs <= DDROP)
    m["lwc_welas"] = np.sum(nw * win * np.pi / 6 * 1000.0 * dobs ** 3, axis=ax) * rhoa * 1e3   # g m-3
    # total water expressed as RH over water (vapour + liquid + ice), for the MBW comparison
    T = np.asarray(v["t"][:])
    p = np.asarray(v["p"][:])
    es = np.exp(54.842763 - 6763.22 / T - 4.210 * np.log(T) + 0.000367 * T
                + np.tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T))
    qs = 0.622 * es / (p - es)
    m["rhw_tot"] = 100 * (m["rh"] + (np.asarray(v["ql"][:]) + np.asarray(v["qi"][:])) / qs)
    dust_n = np.sum(nw[:, 0, :], axis=1) + np.sum(ni[:, 0, :], axis=1)     # mode 1 = dust, per kg
    m["dust_ret"] = dust_n / dust_n[0]
    m["ice_frac_dust"] = np.sum(ni[:, 0, :], axis=1) / dust_n[0]
    m["smax"] = float(np.nanmax(m["rh"]) - 1.0)
    m["nfall_ice"] = np.asarray(v["nfall_ice"][:]) * rhoa / 1e6
    _cache[key] = m
    return m


def windows(o, exp):
    ts = TSAT[exp]
    sd = (o["t"] >= ts) & (o["t"] <= ts + 600)
    si = (o["t"] >= ts + 300) & (o["t"] <= o["tend"])
    return sd, si


def mmean(m, key, tt):
    return float(np.nanmean(np.interp(tt, m["t"], m[key])))


def glaciation_T(t, tg, lwc, tsat):
    """Gas temperature when the 60 s mean LWC falls below 10 % of its maximum
    after cloud onset (end of the liquid phase)."""
    s = smooth(lwc, 61)
    after = t >= tsat - 120
    if not np.any(np.isfinite(s[after])):
        return np.nan
    imax = np.nanargmax(np.where(after, s, -np.inf))
    thr = 0.1 * s[imax]
    later = np.where((np.arange(len(t)) > imax) & (s < thr))[0]
    return float(tg[later[0]]) if len(later) else np.nan


# ----------------------------------------------------------------- the numbers
def numbers():
    R = {"obs": {}, "model": {}, "ratios": {}, "paired": {}, "aerosol": {}}
    O = {e: obs(e) for e in EXPS}
    for e in EXPS:
        o = O[e]
        sd, si = windows(o, e)
        a = initial_psd(e)
        R["aerosol"][e] = {k: a[k] for k in ("S_dust", "S_nacl", "N_dust", "N_nacl", "N_dust_05", "N_nacl_05")}
        pre_nd = float(np.nanmean(o["nd"][o["pre"]]))
        R["obs"][e] = {
            "T0": float(np.nanmean(o["tg"][o["pre"]])), "p0": float(np.nanmean(o["p"][o["pre"]])),
            "Tend": float(np.nanmean(o["tg"][o["t"] > o["tend"] - 30])),
            "RHw0": float(np.nanmean(o["rhw"][o["pre"]])), "RHi0": float(np.nanmean(o["rhi"][o["pre"]])),
            "dTwg": float(np.nanmean((o["tw"] - o["tg"])[o["t"] >= TSAT[e]])),
            "dTwg_pre": float(np.nanmean((o["tw"] - o["tg"])[o["pre"]])),
            "Nd_pre": pre_nd, "Nd_excess": float(np.nanmean(o["nd"][sd]) - pre_nd),
            "Nd_peak_excess": float(np.nanmax(smooth(o["nd"], 31)[sd]) - pre_nd),
            "LWC_max": float(np.nanmax(smooth(o["lwc"], 31)[o["t"] > 0])),
            "LWC_mean": float(np.nanmean(o["lwc"][sd])),
            "Ni": float(np.nanmean(o["ni"][si])), "Ni_peak": float(np.nanmax(smooth(o["ni"], 61)[si])),
            "RHw_max": float(np.nanmax(smooth(o["rhw"], 61)[o["t"] > 0])),
            "T_glac": glaciation_T(o["t"], o["tg"], o["lwc"], TSAT[e]),
            "CPC0": float(np.nanmean(o["cpc"][o["pre"]])),
            "tend": o["tend"], "tsat": TSAT[e],
            "T_tsat": float(np.interp(TSAT[e], o["t"], o["tg"])),
        }
        # Retention of 0.6-2 um particles: last 60 s vs pre-expansion, relative to
        # pure pressure dilution and to the CPC ratio.  In the mixtures most of
        # these are NaCl (KIT split), so this is not a dust-only quantity.
        D = o["D"]
        sel = (D >= 0.6) & (D <= 2.0)
        dlog = np.gradient(np.log10(D))
        n6 = np.nansum(o["psd"][:, sel] * dlog[sel], axis=1)
        late = o["t"] > o["tend"] - 60
        rho = o["p"] / (o["tg"] + 273.15)
        n_ratio = np.nanmean(n6[late]) / np.nanmean(n6[o["pre"]])
        R["obs"][e]["ret_0p6_2_pressure"] = float(n_ratio / (np.nanmean(rho[late]) / np.nanmean(rho[o["pre"]])))
        R["obs"][e]["ret_0p6_2_cpc"] = float(n_ratio / (np.nanmean(o["cpc"][late]) / np.nanmean(o["cpc"][o["pre"]])))
        R["obs"][e]["N_0p6_2_pre"] = float(np.nanmean(n6[o["pre"]]))

    tags = ["fin_m0", "fin_m1", "fin_x1", "fin_m0_dTp03", "fin_m0_dTm03", "fin_m1_dTp03", "fin_m1_dTm03"]
    for tag in tags:
        R["model"][tag] = {}
        for e in EXPS:
            m = model(tag, e)
            if m is None:
                continue
            o = O[e]
            sd, si = windows(o, e)
            td, ti = o["t"][sd], o["t"][si]
            pre_nd = float(np.interp(0.0, m["t"], m["nd"]))
            R["model"][tag][e] = {
                "Nd_excess": mmean(m, "nd", td) - pre_nd, "LWC_mean": mmean(m, "lwc", td),
                "LWC_max": float(np.nanmax(m["lwc"])),
                "Ni": mmean(m, "ni", ti), "Ni13": mmean(m, "ni13", ti), "Ni25": mmean(m, "ni25", ti),
                "Nbig20": mmean(m, "nbig20", ti), "Nbig13": mmean(m, "nbig13", ti),
                "LWC_welas_mean": mmean(m, "lwc_welas", td), "LWC_welas_max": float(np.nanmax(m["lwc_welas"])),
                "RHtot_max": float(np.nanmax(m["rhw_tot"][m["t"] > 0])),
                "RH0": float(100 * m["rh"][0]), "Ni_peak": float(np.nanmax(m["ni"])),
                "Ni_all": mmean(m, "ni_all", ti), "Smax_pct": 100 * m["smax"],
                "T_glac": glaciation_T(m["t"], m["tg"], m["lwc"], TSAT[e]),
                "dust_ret_end": float(m["dust_ret"][-1]),
                "dust_ret_onset": float(np.interp(TSAT[e], m["t"], m["dust_ret"])),
            }
    for tag in tags:
        R["ratios"][tag] = {}
        for e in MIX:
            a, b = R["model"][tag].get(ALONE[e]), R["model"][tag].get(e)
            if a and b:
                R["ratios"][tag][e] = b["Ni"] / a["Ni"]
    R["ratios"]["obs"] = {e: R["obs"][e]["Ni"] / R["obs"][ALONE[e]]["Ni"] for e in MIX}
    R["ratios"]["obs_per_surface"] = {
        e: R["ratios"]["obs"][e] * R["aerosol"][ALONE[e]]["S_dust"] / R["aerosol"][e]["S_dust"] for e in MIX}
    for mode in ("m0", "m1"):
        R["paired"][mode] = {}
        for e in MIX:
            mix, dust = model(f"fin_{mode}", e), model(f"fin_{mode}dust", e)
            if mix is None or dust is None:
                continue
            o = O[e]
            _, si = windows(o, e)
            ti = o["t"][si]
            R["paired"][mode][e] = {
                "Ni_all_mix": mmean(mix, "ni_all", ti), "Ni_all_dust": mmean(dust, "ni_all", ti),
                "P": mmean(mix, "ni_all", ti) / mmean(dust, "ni_all", ti),
                "Ni20_P": mmean(mix, "ni", ti) / mmean(dust, "ni", ti),
                "ret_mix_end": float(mix["dust_ret"][-1]), "ret_dust_end": float(dust["dust_ret"][-1]),
                "Smax_mix": 100 * mix["smax"], "Smax_dust": 100 * dust["smax"],
                "Tglac_mix": glaciation_T(mix["t"], mix["tg"], mix["lwc"], TSAT[e]),
                "Tglac_dust": glaciation_T(dust["t"], dust["tg"], dust["lwc"], TSAT[e]),
                "frozen_dust_frac_mix": float(np.nanmax(mix["ice_frac_dust"] + 0 * mix["t"])),
            }
    # observed n_s ratio over -22 to -30 C (KIT split, CPC dilution) from ns_observed.csv
    ns = {}
    for r in csv.DictReader(open(HERE / "figs" / "ns_observed.csv")):
        if r["dilution"] != "cpc":
            continue
        ns.setdefault(r["exp"], {})[float(r["T_mid_C"])] = _num(r["ns_dust_m2"])
    R["ns_ratio"] = {}
    for e in MIX:
        rat = [ns[e][T] / ns[ALONE[e]][T] for T in ns[e]
               if -30 <= T <= -22 and np.isfinite(ns[e][T]) and np.isfinite(ns[ALONE[e]].get(T, np.nan))]
        R["ns_ratio"][e] = {"median": float(np.median(rat)) if rat else np.nan,
                            "min": float(np.min(rat)) if rat else np.nan, "max": float(np.max(rat)) if rat else np.nan}
        for lo, hi, lab in ((-25, -20, "warm"), (-30, -26, "cold")):
            rr = [ns[e][T] / ns[ALONE[e]][T] for T in ns[e]
                  if lo <= T <= hi and np.isfinite(ns[e][T]) and np.isfinite(ns[ALONE[e]].get(T, np.nan))]
            R["ns_ratio"][e][lab] = [float(np.min(rr)), float(np.max(rr))] if rr else [np.nan, np.nan]
    # Ice ratio using only times when both runs are glaciated (no droplets can be in NIce)
    R["ratio_glaciated"] = {}
    for e in MIX:
        a = ALONE[e]
        Tg = min(R["obs"][e]["T_glac"], R["obs"][a]["T_glac"]) - 0.5
        rr, rs = [], []
        for T in ns[e]:
            if T <= Tg and np.isfinite(ns[e][T]) and np.isfinite(ns[a].get(T, np.nan)):
                rs.append(ns[e][T] / ns[a][T])
        R["ratio_glaciated"][e] = {"T_below": Tg, "ns_ratio_median": float(np.median(rs)) if rs else np.nan,
                                   "n_bins": len(rs),
                                   "ns_ratio_range": [float(np.min(rs)), float(np.max(rs))] if rs else [np.nan, np.nan]}
    for e in EXPS:
        o = O[e]
        T = o["tg"]
        R["obs"][e]["Ni_glaciated_T25_30"] = float(np.nanmean(o["ni"][(T <= -25.5) & (T >= -29.5) & (o["t"] > 0)]))
    return R, O, ns


# --------------------------------------------------------------------- figures
def fig_aerosol(R):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (dust, runs) in zip(axs, (("SDSA01", ["Exp021", "Exp022", "Exp023"]),
                                      ("ATD03", ["Exp025", "Exp026", "Exp027"]))):
        for e in runs:
            a = initial_psd(e)
            ax.loglog(a["d"], np.where(a["dust"] > 0, a["dust"], np.nan), "-", color=C_RUN[e],
                      label=f"{e} dust ({a['N_dust']:.0f} cm$^{{-3}}$)")
            if a["N_nacl"] > 0:
                ax.loglog(a["d"], np.where(a["nacl"] > 0, a["nacl"], np.nan), "--", color=C_RUN[e], lw=1.0,
                          label=f"{e} NaCl ({a['N_nacl']:.0f} cm$^{{-3}}$)")
        ax.set_xlim(0.01, 20)
        ax.set_ylim(1, 3e4)
        ax.set_xlabel("Volume-equivalent diameter (µm)")
        ax.set_title({"SDSA01": "(a) South African soil dust (SDSA01)", "ATD03": "(b) Arizona test dust (ATD03)"}[dust], loc="left")
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(alpha=0.25, which="major", lw=0.4)
    axs[0].set_ylabel("dN/dlogD (cm$^{-3}$)")
    fig.tight_layout()
    save(fig, "fig01_aerosol")


def fig_overview(R, O, ns):
    fig, axs = plt.subplots(4, 2, figsize=(7.0, 7.4), sharex=True)
    groups = (("SDSA01", ["Exp021", "Exp022", "Exp023"]), ("ATD03", ["Exp025", "Exp026", "Exp027"]))
    for j, (dust, runs) in enumerate(groups):
        for e in runs:
            o = O[e]
            c = o["t"] > -30
            lab = e + ("  dust alone" if e not in MIX else "  + NaCl")
            axs[0, j].plot(o["tg"][c], smooth(o["rhw"], 31)[c], color=C_RUN[e], label=lab)
            axs[1, j].plot(o["tg"][c], smooth(o["lwc"], 31)[c], color=C_RUN[e])
            axs[2, j].semilogy(o["tg"][c], smooth(o["ni"], 61)[c], color=C_RUN[e])
            T = np.array(sorted(ns[e]))
            axs[3, j].semilogy(T, [ns[e][x] for x in T], color=C_RUN[e])
        axs[0, j].axhline(100, color=MUTED, lw=0.6)
        axs[0, j].set_title({"SDSA01": "South African soil dust (SDSA01)", "ATD03": "Arizona test dust (ATD03)"}[dust])
        axs[0, j].legend(fontsize=6.5, loc="lower left")
        axs[0, j].set_ylim(80, 118)
        axs[1, j].set_ylim(0, None)
        axs[2, j].set_ylim(5e-3, 5)
        axs[3, j].set_ylim(1e8, 1e11)
        axs[3, j].set_xlabel("Gas temperature (°C)")
        axs[3, j].set_xlim(-15.5, -31)
        for i in range(4):
            axs[i, j].grid(alpha=0.25, lw=0.4)
            axs[i, j].text(0.98, 0.93, "abcdefgh"[2 * i + j], transform=axs[i, j].transAxes,
                           ha="right", va="top", fontweight="bold")
    axs[0, 0].set_ylabel("Total water RH$_w$ (%)\n(MBW)")
    axs[1, 0].set_ylabel("WELAS LWC (g m$^{-3}$)")
    axs[2, 0].set_ylabel("Ice, > 15 µm (cm$^{-3}$)")
    axs[3, 0].set_ylabel("n$_s$ (m$^{-2}$)")
    fig.tight_layout()
    save(fig, "fig02_observed_overview")


def fig_psd(O):
    temps = (-19.0, -22.0, -25.0)
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True, sharey=True)
    groups = (["Exp021", "Exp022", "Exp023"], ["Exp025", "Exp026", "Exp027"])
    for i, runs in enumerate(groups):
        for j, T in enumerate(temps):
            ax = axs[i, j]
            for e in runs:
                o = O[e]
                k = np.where((o["t"] > 0) & (o["tg"] <= T))[0]
                if not len(k):
                    continue
                s = slice(max(k[0] - 15, 0), k[0] + 15)
                psd = np.nanmean(o["psd"][s], axis=0)
                ax.loglog(o["D"], np.where(psd > 0, psd, np.nan), color=C_RUN[e], lw=1.0,
                          label=e + (" + NaCl" if e in MIX else " alone"))
            ax.axvspan(15, 25, color=MUTED, alpha=0.12, lw=0)
            ax.set_xlim(0.39, 150)
            ax.set_ylim(1e-3, 1e4)
            ax.grid(alpha=0.25, lw=0.4)
            ax.set_title(f"({'abcdef'[3 * i + j]}) {T:.0f} °C, {'SDSA01' if i == 0 else 'ATD03'}", loc="left")
            if i == 1:
                ax.set_xlabel("Optical diameter (µm)")
            if j == 0:
                ax.set_ylabel("dN/dlogD (cm$^{-3}$)")
            if j == 2:
                ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    save(fig, "fig03_welas_spectra")


def fig_wall(R):
    warm = {"005": (-0.44, -0.35), "006": (-0.25, -0.09), "007": (-0.35, -0.10), "008": (-0.25, -0.095),
            "009": (-0.25, -0.095), "010": (-0.25, -0.095), "011": (-0.28, -0.10), "013": (-0.27, -0.085),
            "014": (-0.27, -0.106), "015": (-0.26, -0.075), "016": (-0.26, -0.08), "017": (-1.06, -0.55),
            "018": (-0.34, -0.09), "019": (-0.26, -0.10)}
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    x = np.linspace(-1.2, 0.5, 10)
    ax.axvspan(-0.15, 0.15, color=MUTED, alpha=0.15, lw=0)
    ax.plot(x, x / 3, "--", color=MUTED, lw=0.8)
    ax.text(-1.15, -0.33, "ΔT$_{BL}$ = (T$_w$−T$_g$)/3", color=MUTED, fontsize=6.5)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.axvline(0, color=MUTED, lw=0.5)
    for k, (dx, dy) in warm.items():
        mk = "s" if k in ("005", "017") else "o"
        ax.plot(dx, dy, mk, color=INK, ms=4, mfc="white" if k in ("005", "017") else INK)
    for e in EXPS:
        ax.plot(R["obs"][e]["dTwg"], DTBL[e], "D", ms=4.5, color=C_M1 if e not in MIX else C_M0)
        ax.annotate(e[3:], (R["obs"][e]["dTwg"], DTBL[e]), textcoords="offset points", xytext=(3, 3), fontsize=6)
    ax.plot([], [], "o", color=INK, ms=4, label="warm, fan on (Paper 1)")
    ax.plot([], [], "s", color=INK, mfc="white", ms=4, label="warm, Exp005 (fan off), Exp017 (4 K min$^{-1}$)")
    ax.plot([], [], "D", color=C_M1, ms=4, label="cold, dust alone")
    ax.plot([], [], "D", color=C_M0, ms=4, label="cold, dust + NaCl")
    ax.set_xlabel("Measured T$_{wall}$ − T$_{gas}$ (K)")
    ax.set_ylabel("Fitted ΔT$_{BL}$ (K)")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_xlim(-1.2, 0.6)
    ax.set_ylim(-0.6, 0.25)
    fig.tight_layout()
    save(fig, "fig04_wall_offsets")


def fig_liquid(R, O):
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex="col")
    for j, e in enumerate(("Exp021", "Exp025")):
        o = O[e]
        m0, m1 = model("fin_m0", e), model("fin_m1", e)
        c = o["t"] > -30
        pre = np.nanmean(o["nd"][o["pre"]])
        axs[0, j].plot(o["t"][c] / 60, smooth(o["nd"], 31)[c] - pre, color=INK, label="WELAS (minus pre-cloud)")
        axs[1, j].plot(o["t"][c] / 60, smooth(o["lwc"], 31)[c], color=INK, label="WELAS")
        for m, col, lab in ((m0, C_M0, "model, mode 0"), (m1, C_M1, "model, mode 1")):
            if m is None:
                continue
            axs[0, j].plot(m["t"] / 60, m["nd"] - np.interp(0, m["t"], m["nd"]), color=col, label=lab)
            axs[1, j].plot(m["t"] / 60, m["lwc"], color=col, label=lab)
            for sfx, ls in (("_dTp03", ":"), ("_dTm03", ":")):
                ms = model(f"fin_{'m0' if col == C_M0 else 'm1'}{sfx}", e)
                if ms is not None:
                    axs[1, j].plot(ms["t"] / 60, ms["lwc"], ls, color=col, lw=0.8)
        axs[0, j].set_title(f"{e} ({DUST[e]} alone), ΔT$_{{BL}}$ = {DTBL[e]:+.2f} K", loc="left")
        axs[1, j].set_xlabel("Time since start of expansion (min)")
        for i in range(2):
            axs[i, j].grid(alpha=0.25, lw=0.4)
            axs[i, j].axvline(TSAT[e] / 60, color=MUTED, lw=0.6, ls="--")
            axs[i, j].text(0.98, 0.93, "abcd"[2 * i + j], transform=axs[i, j].transAxes, ha="right", va="top",
                           fontweight="bold")
    axs[0, 0].set_ylabel("Droplets 0.39–13 µm (cm$^{-3}$)")
    axs[1, 0].set_ylabel("LWC (g m$^{-3}$)")
    axs[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig05_liquid_dust_alone")


def fig_ice(R, O):
    fig, axs = plt.subplots(3, 2, figsize=(7.0, 6.4), sharex=True, sharey=True)
    order = (("Exp021", "Exp025"), ("Exp022", "Exp026"), ("Exp023", "Exp027"))
    for i, row in enumerate(order):
        for j, e in enumerate(row):
            ax = axs[i, j]
            o = O[e]
            c = o["t"] > 0
            ax.semilogy(o["tg"][c], smooth(o["ni"], 61)[c], color=INK, label="WELAS > 13 µm")
            for mode, col in (("m0", C_M0), ("m1", C_M1)):
                runs = [model(f"fin_{mode}{s}", e) for s in ("", "_dTp03", "_dTm03")]
                if runs[0] is None:
                    continue
                m = runs[0]
                ax.semilogy(m["tg"], m["ni"], color=col, label=f"model, mode {mode[1]}, ice > 20 µm")
                if mode == "m1" and e == "Exp021":
                    ax.semilogy(m["tg"], m["nbig20"], ":", color=col, lw=0.9, label="model, mode 1, ice + drops > 20 µm")
                ok = [r for r in runs if r is not None]
                if len(ok) == 3:
                    lo = np.minimum.reduce([np.interp(m["t"], r["t"], r["ni"]) for r in ok])
                    hi = np.maximum.reduce([np.interp(m["t"], r["t"], r["ni"]) for r in ok])
                    ax.fill_between(m["tg"], np.clip(lo, 1e-4, None), np.clip(hi, 1e-4, None), color=col,
                                    alpha=0.15, lw=0)
            x1 = model("fin_x1", e)
            if x1 is not None:
                ax.semilogy(x1["tg"], x1["ni"], "--", color=C_X1, lw=1.0, label="model, mode 1, X ≥ 0.1")
            a = R["aerosol"][e]
            lab = f"{e}: {DUST[e]}" + (f" + {a['N_nacl']:.0f} cm$^{{-3}}$ NaCl" if e in MIX else " alone")
            ax.set_title(f"({'abcdef'[2 * i + j]}) {lab}", loc="left")
            ax.set_ylim(1e-3, 10)
            ax.set_xlim(-15.5, -31)
            ax.grid(alpha=0.25, lw=0.4)
            if i == 2:
                ax.set_xlabel("Gas temperature (°C)")
            if j == 0:
                ax.set_ylabel("Ice (cm$^{-3}$)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig06_ice_obs_model")


def fig_ratio(R):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    x = np.arange(len(MIX))
    ax.axhline(1, color=MUTED, lw=0.6)
    ax.plot(x - 0.24, [R["ratios"]["obs"][e] for e in MIX], "o", color=INK, ms=6, label="observed")
    ax.plot(x - 0.24, [R["ns_ratio"][e]["median"] for e in MIX], "_", color=INK, ms=10, mew=1.5,
            label="observed, n$_s$ ratio (median −22 to −30 °C)")
    for dx, mode, col, lab in ((-0.06, "m0", C_M0, "model, mode 0 (must activate)"),
                               (0.10, "m1", C_M1, "model, mode 1 (X ≥ 0.01)")):
        y = [R["ratios"][f"fin_{mode}"].get(e, np.nan) for e in MIX]
        sens = [[R["ratios"][f"fin_{mode}{s}"].get(e, np.nan) for s in ("_dTp03", "_dTm03")] for e in MIX]
        lo = [min([y[k]] + sens[k]) for k in range(len(MIX))]
        hi = [max([y[k]] + sens[k]) for k in range(len(MIX))]
        ax.errorbar(x + dx, y, yerr=[np.array(y) - lo, np.array(hi) - y], fmt="s", color=col, ms=5,
                    capsize=2, lw=1, label=lab)
    ax.plot(x + 0.24, [R["ratios"]["fin_x1"].get(e, np.nan) for e in MIX], "^", color=C_X1, ms=5,
            label="model, mode 1, X ≥ 0.1")
    ax.set_yscale("log")
    ax.set_ylim(0.005, 5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\n÷ {ALONE[e]}" for e in MIX])
    ax.set_ylabel("Ice, mixture ÷ dust alone")
    ax.set_title("(a) Mixture ÷ dust alone (obs. and model)", loc="left")
    ax.legend(fontsize=6, loc="lower left", ncol=1)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    ax = axs[1]
    ax.axhline(1, color=MUTED, lw=0.6)
    for dx, mode, col, lab in ((-0.1, "m0", C_M0, "mode 0"), (0.1, "m1", C_M1, "mode 1")):
        ax.plot(x + dx, [R["paired"][mode].get(e, {}).get("P", np.nan) for e in MIX], "s", color=col, ms=5, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 50)
    ax.set_xticks(x)
    ax.set_xticklabels(MIX)
    ax.set_ylabel("Ice with NaCl ÷ without")
    ax.set_title("(b) Paired runs (model only)", loc="left")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    fig.tight_layout()
    save(fig, "fig07_ratios")


def fig_mechanisms(R, O):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
    ax = axs[0]
    for e, col in (("Exp026", "#2a78d6"), ("Exp027", "#eb6834"), ("Exp022", "#1baf7a"), ("Exp023", "#e87ba4")):
        mix, dust = model("fin_m1", e), model("fin_m1dust", e)
        if mix is None or dust is None:
            continue
        ax.plot(mix["tg"], mix["dust_ret"], "-", color=col, label=f"{e}")
        ax.plot(dust["tg"], dust["dust_ret"], ":", color=col)
    ax.plot([], [], "-", color=MUTED, label="with NaCl")
    ax.plot([], [], ":", color=MUTED, label="NaCl removed")
    ax.set_xlim(-15.5, -31)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Gas temperature (°C)")
    ax.set_ylabel("Dust still airborne (fraction)")
    ax.set_title("(a) Model, mode 1", loc="left")
    ax.legend(fontsize=5.8, loc="lower left")
    ax.grid(alpha=0.25, lw=0.4)
    ax = axs[1]
    for e, col in (("Exp026", "#2a78d6"), ("Exp027", "#eb6834"), ("Exp022", "#1baf7a"), ("Exp023", "#e87ba4")):
        mix, dust = model("fin_m1", e), model("fin_m1dust", e)
        if mix is None or dust is None:
            continue
        ax.plot(mix["tg"], 100 * (mix["rh"] - 1), "-", color=col, lw=1.0)
        ax.plot(dust["tg"], 100 * (dust["rh"] - 1), ":", color=col, lw=1.0)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlim(-15.5, -31)
    ax.set_ylim(-3, 0.6)
    ax.set_xlabel("Gas temperature (°C)")
    ax.set_ylabel("Supersaturation w.r.t. water (%)")
    ax.set_title("(b) Model, mode 1", loc="left")
    ax.grid(alpha=0.25, lw=0.4)
    ax = axs[2]
    x = np.arange(len(EXPS))
    ax.plot(x - 0.12, [R["obs"][e]["T_glac"] for e in EXPS], "o", color=INK, ms=5, label="observed")
    ax.plot(x + 0.12, [R["model"]["fin_m1"].get(e, {}).get("T_glac", np.nan) for e in EXPS], "s", color=C_M1,
            ms=5, label="model, mode 1")
    ax.set_xticks(x)
    ax.set_xticklabels([e[3:] for e in EXPS])
    ax.set_xlabel("Experiment")
    ax.set_ylabel("End of liquid phase (°C)")
    ax.set_title("(c) Glaciation", loc="left")
    ax.invert_yaxis()
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    save(fig, "fig08_mechanisms")


def fig_schematic():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0))
    titles = ("(a) Dust alone", "(b) + NaCl, dust must activate\n(mode 0): suppression",
              "(c) + NaCl, wetted dust can freeze\n(mode 1): no suppression")
    for ax, t in zip(axs, titles):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(t, fontsize=7.5, loc="left")
    dust_c, salt_c, drop_c, ice_c = "#8c6d46", "#bfbfbf", "#9ec5f4", "#2a78d6"

    def dust(ax, x, y, r=0.25):
        ax.add_patch(Circle((x, y), r, color=dust_c, zorder=3))

    def drop(ax, x, y, r, c=drop_c, a=0.8):
        ax.add_patch(Circle((x, y), r, color=c, alpha=a, zorder=2))

    def crystal(ax, x, y, s=0.55):
        for ang in (0, 60, 120):
            th = np.deg2rad(ang)
            ax.plot([x - s * np.cos(th), x + s * np.cos(th)], [y - s * np.sin(th), y + s * np.sin(th)],
                    color=ice_c, lw=1.6, zorder=4)

    def arrow(ax, x0, y0, x1, y1, txt=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=MUTED, lw=0.9))
        if txt:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.35, txt, fontsize=6, ha="center", color=INK)

    # (a) dust alone: activates, droplets grow, freeze, some sediment
    ax = axs[0]
    ax.text(0.2, 9.0, "S$_w$ > 0: dust activates", fontsize=6.3)
    dust(ax, 1.2, 7.4)
    arrow(ax, 1.7, 7.4, 3.4, 7.4)
    drop(ax, 4.4, 7.4, 0.8)
    dust(ax, 4.4, 7.4)
    arrow(ax, 5.4, 7.4, 7.0, 7.4, "freezes")
    crystal(ax, 8.2, 7.4)
    dust(ax, 8.2, 7.4, 0.18)
    drop(ax, 4.4, 3.8, 1.0)
    dust(ax, 4.4, 3.8)
    arrow(ax, 4.4, 2.6, 4.4, 0.6)
    ax.text(4.9, 1.4, "large drops and ice\nfall out: dust lost", fontsize=6)
    ax.text(0.2, 5.4, "few, large drops;\nwater-saturated cloud", fontsize=6)
    # (b) mode 0: NaCl takes the vapour; dust stays dry; no ice
    ax = axs[1]
    ax.text(0.2, 9.0, "NaCl haze holds S$_w$ < S$_{crit}$(dust)", fontsize=6.3)
    for (x, y) in ((1.0, 7.3), (2.4, 6.4), (3.4, 7.8), (6.6, 7.6), (8.2, 6.6), (7.6, 8.4), (1.6, 4.6), (8.8, 4.8)):
        drop(ax, x, y, 0.42, c=salt_c, a=0.6)
    dust(ax, 5.0, 6.6)
    ax.text(5.0, 5.4, "dust not activated", fontsize=6, ha="center")
    ax.text(5.0, 4.5, "→ no immersion freezing", fontsize=6, ha="center")
    ax.text(5.0, 2.2, "model: ice ÷ 9 to ÷ 70\nobserved: ÷ 1.3 to ÷ 2", fontsize=6.3, ha="center", color=C_M0)
    # (c) mode 1: wetted dust freezes in haze; stays airborne
    ax = axs[2]
    ax.text(0.2, 9.0, "S$_w$ just below 0; FHH water film on dust", fontsize=6.3)
    for (x, y) in ((1.0, 7.3), (2.6, 8.0), (7.0, 8.2), (8.6, 6.8), (1.4, 4.6), (8.8, 4.6)):
        drop(ax, x, y, 0.42, c=salt_c, a=0.6)
    drop(ax, 4.0, 6.6, 0.38)
    dust(ax, 4.0, 6.6)
    arrow(ax, 4.6, 6.6, 6.2, 6.6, "freezes")
    crystal(ax, 7.1, 6.0)
    dust(ax, 7.1, 6.0, 0.18)
    ax.text(5.0, 4.1, "dust stays small and airborne\nuntil colder: more can freeze", fontsize=6, ha="center")
    ax.text(5.0, 2.0, "model (paired): ice × 2 to × 14\nobserved: ÷ 1.3 to ÷ 2", fontsize=6.3, ha="center", color=C_M1)
    fig.tight_layout()
    save(fig, "fig09_schematic")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def main():
    R, O, ns = numbers()
    with open(OUT / "numbers.json", "w") as fh:
        json.dump(R, fh, indent=1, default=float)
    for f in (fig_aerosol, ):
        f(R)
    fig_overview(R, O, ns)
    fig_psd(O)
    fig_wall(R)
    fig_liquid(R, O)
    fig_ice(R, O)
    fig_ratio(R)
    fig_mechanisms(R, O)
    fig_schematic()
    print("written", OUT)


if __name__ == "__main__":
    main()
