"""Figures and numbers for the iSKYLAB mixed-phase paper, draft v0.2 (16 Nov 2026).

Final analysis design (see claude/iskylab_mixed_phase_paper_plan.md):
  * n_s(T) of each dust fitted to its dust-alone experiment (BMM categories
    sdsa01_iskylab / atd03_iskylab; runs tagged ns_m<mode>_<offset>);
  * wall offset dT_BL fitted to the liquid phase only (refit_score.py), within
    the physical range [min(0, Tw-Tg) - 0.15 K, max(0, Tw-Tg) + 0.15 K];
  * mode 0: dust must activate to freeze; mode 1: dust freezes once its
    adsorbed water is >= 1 % of its dry volume.
Inputs: paper/refit_choice_ns_constrained.txt, paper/refit_scores_ns_constrained.csv,
        figs/ns_observed.csv, the KIT data files and the BMM runs.
Outputs: paper/figs/f*.pdf|png and paper/numbers_v2.json.

Usage: python3 paper_figs_v2.py      (run refit_score.py --prefix ns --constrain first)
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

import paper_figs as P

OUT, FIG = P.OUT, P.FIG
EXPS, MIX, ALONE, DUST = P.EXPS, P.MIX, P.ALONE, P.DUST
INK, MUTED = P.INK, P.MUTED
C0, C1 = "#2a78d6", "#eb6834"          # mode 0 blue, mode 1 orange
CMIX = {"Exp021": INK, "Exp025": INK, "Exp022": "#2a78d6", "Exp026": "#2a78d6",
        "Exp023": "#eb6834", "Exp027": "#eb6834"}
ACCEPT = 0.3
NAME = {"SDSA01": "South African soil dust (SDSA01)", "ATD03": "Arizona test dust (ATD03)"}
WARM = {"005": (-0.44, -0.35), "006": (-0.25, -0.09), "007": (-0.35, -0.10), "008": (-0.25, -0.095),
        "009": (-0.25, -0.095), "010": (-0.25, -0.095), "011": (-0.28, -0.10), "013": (-0.27, -0.085),
        "014": (-0.27, -0.106), "015": (-0.26, -0.075), "016": (-0.26, -0.08), "017": (-1.06, -0.55),
        "018": (-0.34, -0.09), "019": (-0.26, -0.10)}
NS_FIT = {"SDSA01": (-0.7640, 2.634, -27.75, 0.517), "ATD03": (-1.5437, -13.511, -24.75, 0.380)}
NS_PUB = {"SDSA01": (lambda t: np.exp(-0.517 * t + 8.934), (-36, -12), "Niemand et al. (2012), desert dust"),
          "ATD03": (lambda t: np.exp(-0.380 * t + 13.918), (-26.7, -17.7), "published ATD fit")}


def tagof(mode, off):
    v = round(off * 1000)
    return f"ns_m{mode}_{'m' if v < 0 else 'p'}{abs(v):03d}"


def ns_iskylab(dust, t, flat=False):
    b, a, tc, slope = NS_FIT[dust]
    t = np.asarray(t, float)
    ns = np.exp(b * np.maximum(t, tc) + a)
    if not flat:
        ns = ns * np.exp(-slope * np.minimum(t - tc, 0.0))
    return np.where(t > -18, np.nan, ns)


def load_choice():
    ch = {}
    for line in open(OUT / "refit_choice_ns_constrained.txt"):
        if line.startswith("#") or not line.strip():
            continue
        e, m, off = line.split()
        ch[(e, int(m))] = float(off)
    rows = list(csv.DictReader(open(OUT / "refit_scores_ns_constrained.csv")))
    acc = {}
    for (e, m), off in ch.items():
        o = P.obs(e)
        dtw = float(np.nanmean((o["tw"] - o["tg"])[o["t"] >= P.TSAT[e]]))
        lo, hi = min(0.0, dtw) - 0.15, max(0.0, dtw) + 0.15
        cand = [r for r in rows if r["exp"] == e and int(r["mode"]) == m and lo - 1e-9 <= float(r["dTBL_K"]) <= hi + 1e-9]
        best = min(float(r["score"]) for r in cand)
        acc[(e, m)] = sorted({r["tag"] for r in cand if float(r["score"]) <= best + ACCEPT})
    return ch, acc, rows


def ice_mean(m, o, e, key="ni"):
    ti = o["t"][P.windows(o, e)[1]]
    return P.mmean(m, key, ti)


# ------------------------------------------------------------------ numbers
def numbers(ch, acc):
    R = {"choice": {f"{e}_m{m}": off for (e, m), off in ch.items()}, "obs": {}, "model": {}, "ratio": {}, "paired": {},
         "accept": {f"{e}_m{m}": v for (e, m), v in acc.items()}}
    O = {e: P.obs(e) for e in EXPS}
    for e in EXPS:
        o = O[e]
        sd, si = P.windows(o, e)
        R["obs"][e] = {"dTwg": float(np.nanmean((o["tw"] - o["tg"])[o["t"] >= P.TSAT[e]])),
                       "Nd": float(np.nanmean(o["nd"][sd]) - np.nanmean(o["nd"][o["pre"]])),
                       "LWC": float(np.nanmean(o["lwc"][sd])), "TW": float(np.nanmean(o["rhw"][sd]) - 100),
                       "Ni": float(np.nanmean(o["ni"][si])), "Ni_peak": float(np.nanmax(P.smooth(o["ni"], 61)[o["t"] > 0])),
                       "T_peak": float(o["tg"][np.nanargmax(np.where(o["t"] > 0, P.smooth(o["ni"], 61), -1))]),
                       "T_glac": P.glaciation_T(o["t"], o["tg"], o["lwc"], P.TSAT[e])}
        for mode in (0, 1):
            tag = tagof(mode, ch[(e, mode)])
            m = P.model(tag, e)
            td = o["t"][sd]
            pk = int(np.argmax(m["ni"]))
            R["model"][f"{e}_m{mode}"] = {
                "tag": tag, "dTBL": ch[(e, mode)],
                "Nd": P.mmean(m, "nd", td) - float(np.interp(0, m["t"], m["nd"])),
                "LWC": P.mmean(m, "lwc_welas", td), "LWC_true": P.mmean(m, "lwc", td),
                "TW": P.mmean(m, "rhw_tot", td) - 100, "Ni": ice_mean(m, o, e), "Ni_all": ice_mean(m, o, e, "ni_all"),
                "Ni13": ice_mean(m, o, e, "ni13"), "Ni25": ice_mean(m, o, e, "ni25"),
                "Ni_peak": float(m["ni"][pk]), "T_peak": float(m["tg"][pk]),
                "T_glac": P.glaciation_T(m["t"], m["tg"], m["lwc"], P.TSAT[e]),
                "Ni_range": [min(ice_mean(P.model(t, e), o, e) for t in acc[(e, mode)]),
                             max(ice_mean(P.model(t, e), o, e) for t in acc[(e, mode)])]}
    for e in MIX:
        a = ALONE[e]
        R["ratio"][e] = {"obs": R["obs"][e]["Ni"] / R["obs"][a]["Ni"]}
        surf = P.initial_psd(a)["S_dust"] / P.initial_psd(e)["S_dust"]
        R["ratio"][e]["obs_per_surface"] = R["ratio"][e]["obs"] * surf
        for mode in (0, 1):
            best = R["model"][f"{e}_m{mode}"]["Ni"] / R["model"][f"{a}_m{mode}"]["Ni"]
            rs = [ice_mean(P.model(x, e), O[e], e) / ice_mean(P.model(y, a), O[a], a)
                  for x in acc[(e, mode)] for y in acc[(a, mode)]]
            R["ratio"][e][f"m{mode}"] = best
            R["ratio"][e][f"m{mode}_range"] = [min(rs), max(rs)]
            tag = tagof(mode, ch[(e, mode)])
            mix, dus = P.model(tag, e), P.model(tag + "dust", e)
            if dus is not None:
                R["paired"][f"{e}_m{mode}"] = {
                    "P": ice_mean(mix, O[e], e, "ni_all") / ice_mean(dus, O[e], e, "ni_all"),
                    "P20": ice_mean(mix, O[e], e) / max(ice_mean(dus, O[e], e), 1e-12),
                    "ret_mix": float(mix["dust_ret"][-1]), "ret_dust": float(dus["dust_ret"][-1]),
                    "Tglac_mix": P.glaciation_T(mix["t"], mix["tg"], mix["lwc"], P.TSAT[e]),
                    "Tglac_dust": P.glaciation_T(dus["t"], dus["tg"], dus["lwc"], P.TSAT[e])}
    # n_s calibration: published fit / observed for the dust-alone runs
    ns = {}
    for r in csv.DictReader(open(P.HERE / "figs" / "ns_observed.csv")):
        if r["dilution"] == "cpc":
            ns.setdefault(r["exp"], {})[float(r["T_mid_C"])] = P._num(r["ns_dust_m2"])
    R["ns_obs_over_pub"] = {}
    for e, d in (("Exp021", "SDSA01"), ("Exp025", "ATD03")):
        f = NS_PUB[d][0]
        R["ns_obs_over_pub"][e] = {f"{T:.2f}": ns[e][T] / f(T) for T in sorted(ns[e]) if -30 <= T <= -18 and np.isfinite(ns[e][T])}
    # flat-ns sensitivity (mode 1 at the chosen offsets)
    R["flat"] = {}
    for e in EXPS:
        m = P.model("nsf" + tagof(1, ch[(e, 1)])[2:], e)
        if m is not None:
            R["flat"][e] = ice_mean(m, O[e], e)
    return R, O, ns


# ------------------------------------------------------------------ figures
def style(ax, letter=None):
    ax.grid(alpha=0.25, lw=0.4)
    if letter:
        ax.text(0.02, 0.97, letter, transform=ax.transAxes, ha="left", va="top", fontweight="bold")


def tx(ax):
    ax.set_xlim(-15.5, -31)


def fig_ns(ns):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, (e, d) in zip(axs, (("Exp021", "SDSA01"), ("Exp025", "ATD03"))):
        T = np.array(sorted(ns[e]))
        y = np.array([ns[e][t] for t in T])
        ok = np.isfinite(y) & (T >= -30.5)
        ax.semilogy(T[ok], y[ok], "o", ms=3.2, color=INK, label=f"observed, {e}")
        f, (lo, hi), lab = NS_PUB[d]
        tt = np.linspace(lo, hi, 50)
        ax.semilogy(tt, f(tt), "--", color=MUTED, lw=1.2, label=lab)
        tt = np.linspace(-18, -31, 120)
        ax.semilogy(tt, ns_iskylab(d, tt), "-", color=C1, lw=1.8, label="fit used here")
        ax.semilogy(tt, ns_iskylab(d, tt, flat=True), ":", color=C1, lw=1.2, label="fit held constant (test)")
        b, a, tc, _ = NS_FIT[d]
        ax.axvspan(tc, -20.75 if d == "SDSA01" else -21.75, color=C1, alpha=0.08, lw=0)
        ax.set_ylim(1e6, 3e11)
        ax.set_xlim(-17.5, -31)
        ax.set_xlabel("Gas temperature (°C)")
        ax.set_title(NAME[d], loc="left")
        ax.legend(fontsize=6.3, loc="lower right")
        style(ax, "ab"[0 if d == "SDSA01" else 1])
    axs[0].set_ylabel("n$_s$ (m$^{-2}$)")
    fig.tight_layout()
    P.save(fig, "f05_ns_calibration")


def fig_wall(R):
    fig, ax = plt.subplots(figsize=(3.7, 3.1))
    x = np.linspace(-1.2, 0.6, 10)
    ax.fill_between(x, np.minimum(0, x) - 0.15, np.maximum(0, x) + 0.15, color=MUTED, alpha=0.15, lw=0,
                    label="between gas and wall, ±0.15 K")
    ax.plot(x, x / 3, "--", color=MUTED, lw=0.8, label="one third of T$_w$−T$_g$")
    ax.axhline(0, color=MUTED, lw=0.5)
    for k, (dx, dy) in WARM.items():
        special = k in ("005", "017")
        ax.plot(dx, dy, "s" if special else "o", color=INK, mfc="white" if special else INK, ms=4)
    for e in EXPS:
        ax.plot(R["obs"][e]["dTwg"], R["choice"][f"{e}_m1"], "D", ms=5, color=C1, zorder=4)
        off = {"Exp025": (5, 4), "Exp026": (-14, 6), "Exp027": (5, -9)}.get(e, (5, -2))
        ax.annotate(e[3:], (R["obs"][e]["dTwg"], R["choice"][f"{e}_m1"]), textcoords="offset points",
                    xytext=off, fontsize=6)
    ax.plot([], [], "o", color=INK, ms=4, label="warm (Paper 1)")
    ax.plot([], [], "s", color=INK, mfc="white", ms=4, label="warm, Exp005 and Exp017")
    ax.plot([], [], "D", color=C1, ms=5, label="cold, this study")
    ax.set_xlabel("Measured T$_{wall}$ − T$_{gas}$ (K)")
    ax.set_ylabel("Fitted ΔT$_{BL}$ (K)")
    ax.set_xlim(-1.15, 0.55)
    ax.set_ylim(-0.6, 0.6)
    ax.legend(fontsize=5.8, loc="upper left")
    style(ax)
    fig.tight_layout()
    P.save(fig, "f04_wall_offsets")


def fig_liquid_alone(R, O):
    fig, axs = plt.subplots(3, 2, figsize=(7.0, 6.0), sharex=True)
    for j, e in enumerate(("Exp021", "Exp025")):
        o = O[e]
        c = o["t"] > 0
        pre = np.nanmean(o["nd"][o["pre"]])
        axs[0, j].plot(o["tg"][c], P.smooth(o["nd"], 31)[c] - pre, color=INK, label="observed")
        axs[1, j].plot(o["tg"][c], P.smooth(o["lwc"], 31)[c], color=INK)
        axs[2, j].plot(o["tg"][c], P.smooth(o["rhw"], 31)[c] - 100, color=INK)
        for mode, col in ((0, C0), (1, C1)):
            m = P.model(tagof(mode, R["choice"][f"{e}_m{mode}"]), e)
            lab = f"model, mode {mode}"
            axs[0, j].plot(m["tg"], m["nd"] - np.interp(0, m["t"], m["nd"]), color=col, label=lab)
            axs[1, j].plot(m["tg"], m["lwc_welas"], color=col)
            axs[2, j].plot(m["tg"], m["rhw_tot"] - 100, color=col)
        axs[0, j].set_title(f"{e}: {DUST[e]} alone, ΔT$_{{BL}}$ = {R['choice'][e + '_m1']:+.2f} K", loc="left")
        axs[2, j].set_xlabel("Gas temperature (°C)")
        for i in range(3):
            tx(axs[i, j])
            style(axs[i, j], "abcdef"[2 * i + j])
        axs[2, j].axhline(0, color=MUTED, lw=0.6)
    axs[0, 0].set_ylabel("Droplets above\npre-cloud (cm$^{-3}$)")
    axs[1, 0].set_ylabel("Liquid water seen\nby WELAS (g m$^{-3}$)")
    axs[2, 0].set_ylabel("Total water above\nwater saturation (%)")
    axs[0, 0].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    P.save(fig, "f06_liquid_dust_alone")


def fig_liquid_mix(R, O):
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.8), sharex=True, sharey="row")
    for j, e in enumerate(("Exp022", "Exp023", "Exp026", "Exp027")):
        o = O[e]
        c = o["t"] > 0
        pre = np.nanmean(o["nd"][o["pre"]])
        axs[0, j].plot(o["tg"][c], P.smooth(o["rhw"], 31)[c] - 100, color=INK, label="observed")
        axs[1, j].plot(o["tg"][c], P.smooth(o["nd"], 31)[c] - pre, color=INK)
        for mode, col in ((0, C0), (1, C1)):
            m = P.model(tagof(mode, R["choice"][f"{e}_m{mode}"]), e)
            axs[0, j].plot(m["tg"], m["rhw_tot"] - 100, color=col, label=f"model, mode {mode}")
            axs[1, j].plot(m["tg"], m["nd"] - np.interp(0, m["t"], m["nd"]), color=col)
        axs[0, j].set_title(f"{e} ({'SDSA01' if DUST[e] == 'SDSA01' else 'ATD03'})", loc="left", fontsize=7.5)
        axs[1, j].set_xlabel("Gas T (°C)")
        for i in range(2):
            tx(axs[i, j])
            style(axs[i, j], "abcdefgh"[4 * i + j])
        axs[0, j].axhline(0, color=MUTED, lw=0.6)
    axs[0, 0].set_ylabel("Total water above\nwater saturation (%)")
    axs[1, 0].set_ylabel("Droplets above\npre-cloud (cm$^{-3}$)")
    axs[1, 0].set_ylim(-150, 900)
    axs[0, 0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    P.save(fig, "f07_liquid_mixtures")


def fig_ice(R, O, acc):
    fig, axs = plt.subplots(3, 2, figsize=(7.0, 6.6), sharex=True, sharey=True)
    order = (("Exp021", "Exp025"), ("Exp022", "Exp026"), ("Exp023", "Exp027"))
    for i, row in enumerate(order):
        for j, e in enumerate(row):
            ax = axs[i, j]
            o = O[e]
            c = o["t"] > 0
            ax.semilogy(o["tg"][c], P.smooth(o["ni"], 61)[c], color=INK, lw=1.6, label="observed (WELAS, > 13 µm)")
            for mode, col in ((0, C0), (1, C1)):
                m = P.model(tagof(mode, R["choice"][f"{e}_m{mode}"]), e)
                ax.semilogy(m["tg"], np.clip(m["ni"], 1e-4, None), color=col, label=f"model, mode {mode} (> 20 µm)")
                runs = [P.model(t, e) for t in acc[(e, mode)]]
                if len(runs) > 1:
                    arr = np.array([np.interp(m["t"], r["t"], r["ni"]) for r in runs])
                    ax.fill_between(m["tg"], np.clip(arr.min(0), 1e-4, None), np.clip(arr.max(0), 1e-4, None),
                                    color=col, alpha=0.15, lw=0)
            a = P.initial_psd(e)
            lab = f"{e}: {DUST[e]}" + (f" + {a['N_nacl']:.0f} cm$^{{-3}}$ NaCl" if e in MIX else " alone")
            ax.set_title(f"({'abcdef'[2 * i + j]}) {lab}", loc="left", fontsize=7.5)
            ax.set_ylim(1e-3, 30)
            tx(ax)
            ax.grid(alpha=0.25, lw=0.4)
            if i == 2:
                ax.set_xlabel("Gas temperature (°C)")
            if j == 0:
                ax.set_ylabel("Ice (cm$^{-3}$)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    P.save(fig, "f08_ice")


def fig_ratio(R):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    x = np.arange(len(MIX))
    ax.axhline(1, color=MUTED, lw=0.6)
    ax.plot(x - 0.2, [R["ratio"][e]["obs"] for e in MIX], "o", color=INK, ms=6, label="observed")
    ax.plot(x - 0.2, [R["ratio"][e]["obs_per_surface"] for e in MIX], "o", color=INK, mfc="white", ms=6,
            label="observed, per unit dust surface")
    for dx, mode, col in ((0.0, 0, C0), (0.2, 1, C1)):
        y = np.array([max(R["ratio"][e][f"m{mode}"], 2e-3) for e in MIX])
        lo = np.array([max(R["ratio"][e][f"m{mode}_range"][0], 2e-3) for e in MIX])
        hi = np.array([max(R["ratio"][e][f"m{mode}_range"][1], 2e-3) for e in MIX])
        ax.errorbar(x + dx, y, yerr=[y - lo, hi - y], fmt="s", color=col, ms=5, capsize=2, lw=1,
                    label=f"model, mode {mode}" + (" (must activate)" if mode == 0 else " (wetted dust freezes)"))
    ax.set_yscale("log")
    ax.set_ylim(1.5e-3, 4)
    ax.text(3.35, 2.1e-3, "mode 0 values ≤ 0.002\nplotted at the axis", fontsize=5.8, ha="right", color=C0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\n÷ {ALONE[e]}" for e in MIX], fontsize=7)
    ax.set_ylabel("Ice, mixture ÷ dust alone")
    ax.set_title("(a) Observed and simulated, same definition", loc="left", fontsize=7.5)
    ax.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    ax = axs[1]
    ax.axhline(1, color=MUTED, lw=0.6)
    for dx, mode, col in ((-0.1, 0, C0), (0.1, 1, C1)):
        ax.plot(x + dx, [max(R["paired"][f"{e}_m{mode}"]["P"], 2e-3) for e in MIX], "s", color=col, ms=5,
                label=f"mode {mode}")
    ax.set_yscale("log")
    ax.set_ylim(1.5e-3, 30)
    ax.set_xticks(x)
    ax.set_xticklabels(MIX, fontsize=7)
    ax.set_ylabel("Ice with NaCl ÷ same run without NaCl")
    ax.set_title("(b) Paired simulations (model only)", loc="left", fontsize=7.5)
    ax.legend(fontsize=6.5, loc="center left")
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    fig.tight_layout()
    P.save(fig, "f09_ratios")


def fig_mech(R, O):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    cols = {"Exp022": "#1baf7a", "Exp023": "#e87ba4", "Exp026": "#2a78d6", "Exp027": "#eb6834"}
    for e in MIX:
        tag = tagof(1, R["choice"][f"{e}_m1"])
        mix, dus = P.model(tag, e), P.model(tag + "dust", e)
        ax.plot(mix["tg"], mix["dust_ret"], "-", color=cols[e], label=e)
        ax.plot(dus["tg"], dus["dust_ret"], ":", color=cols[e])
    ax.plot([], [], "-", color=MUTED, label="with NaCl")
    ax.plot([], [], ":", color=MUTED, label="NaCl removed")
    tx(ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Gas temperature (°C)")
    ax.set_ylabel("Fraction of dust still airborne")
    ax.set_title("(a) Model, mode 1", loc="left", fontsize=7.5)
    ax.legend(fontsize=6, loc="lower left", ncol=2)
    ax.grid(alpha=0.25, lw=0.4)
    ax = axs[1]
    x = np.arange(len(EXPS))
    ax.plot(x - 0.15, [R["obs"][e]["T_glac"] for e in EXPS], "o", color=INK, ms=5, label="observed")
    ax.plot(x, [R["model"][f"{e}_m1"]["T_glac"] for e in EXPS], "s", color=C1, ms=5, label="model, mode 1")
    ax.plot([x[MIX.index(e) and 0] for e in []], [], "")
    xm = [EXPS.index(e) + 0.15 for e in MIX]
    ax.plot(xm, [R["paired"][f"{e}_m1"]["Tglac_dust"] for e in MIX], "s", color=C1, mfc="white", ms=5,
            label="model, mode 1, NaCl removed")
    ax.set_xticks(x)
    ax.set_xticklabels([e[3:] for e in EXPS])
    ax.set_xlabel("Experiment")
    ax.set_ylabel("End of the liquid phase (°C)")
    ax.set_ylim(-23.5, -30.5)
    ax.set_title("(b) Glaciation", loc="left", fontsize=7.5)
    ax.legend(fontsize=6, loc="lower left")
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    P.save(fig, "f10_mechanisms")


def fig_schematic(R):
    m0 = max(R["ratio"][e]["m0_range"][1] for e in MIX)
    p1 = [R["paired"][f"{e}_m1"]["P"] for e in MIX]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0))
    titles = ("(a) Dust alone", "(b) + NaCl, if dust must activate\n(mode 0): ice suppressed",
              "(c) + NaCl, if wetted dust can freeze\n(mode 1): ice not suppressed")
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

    ax = axs[0]
    ax.text(0.2, 9.0, "supersaturated: dust activates", fontsize=6.3)
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
    ax.text(4.9, 1.4, "large drops fall out,\ntaking their dust", fontsize=6)
    ax = axs[1]
    ax.text(0.2, 9.0, "NaCl takes the vapour; S$_w$ stays ≈ 0", fontsize=6.3)
    for (x, y) in ((1.0, 7.3), (2.4, 6.4), (3.4, 7.8), (6.6, 7.6), (8.2, 6.6), (7.6, 8.4), (1.6, 4.6), (8.8, 4.8)):
        drop(ax, x, y, 0.42, c=salt_c, a=0.6)
    dust(ax, 5.0, 6.6)
    ax.text(5.0, 5.4, "dust does not activate", fontsize=6, ha="center")
    ax.text(5.0, 4.5, "→ cannot freeze", fontsize=6, ha="center")
    ax.text(5.0, 2.2, f"model: ice × {m0:.2f} or less\nobserved: × 0.50–0.79", fontsize=6.3, ha="center", color=C0)
    ax = axs[2]
    ax.text(0.2, 9.0, "adsorbed water film on the dust", fontsize=6.3)
    for (x, y) in ((1.0, 7.3), (2.6, 8.0), (7.0, 8.2), (8.6, 6.8), (1.4, 4.6), (8.8, 4.6)):
        drop(ax, x, y, 0.42, c=salt_c, a=0.6)
    drop(ax, 4.0, 6.6, 0.38)
    dust(ax, 4.0, 6.6)
    arrow(ax, 4.6, 6.6, 6.2, 6.6, "freezes")
    crystal(ax, 7.1, 6.0)
    dust(ax, 7.1, 6.0, 0.18)
    ax.text(5.0, 4.1, "the dust freezes without activating,\nand stays small and airborne", fontsize=6, ha="center")
    lo = min(R["ratio"][e]["m1_range"][0] for e in MIX)
    hi = max(R["ratio"][e]["m1_range"][1] for e in MIX)
    ax.text(5.0, 2.0, f"model: ice × {lo:.2f}–{hi:.2f}\nobserved: × 0.50–0.79", fontsize=6.3, ha="center", color=C1)
    fig.tight_layout()
    P.save(fig, "f11_schematic")


def main():
    ch, acc, rows = load_choice()
    R, O, ns = numbers(ch, acc)
    with open(OUT / "numbers_v2.json", "w") as fh:
        json.dump(R, fh, indent=1, default=float)
    # Figures 1-3 (aerosol, observed overview, WELAS spectra) come from paper_figs.py.
    fig_wall(R)
    fig_ns(ns)
    fig_liquid_alone(R, O)
    fig_liquid_mix(R, O)
    fig_ice(R, O, acc)
    fig_ratio(R)
    fig_mech(R, O)
    fig_schematic(R)
    print("written", OUT / "numbers_v2.json", "and figures f04-f11 in", FIG)


if __name__ == "__main__":
    main()
