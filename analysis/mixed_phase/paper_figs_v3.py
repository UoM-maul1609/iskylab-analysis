"""Figures and numbers for the iSKYLAB mixed-phase paper, draft v0.3 (17 Nov 2026).

Analysis design (claude/iskylab_mixed_phase_paper_plan.md):
  * runs wk3c_m<mode>_<offset>: measured start humidity, aerosol at t = 0, wall vapour
    sink km = 3e-4 m/s, n_s from the dust-alone runs (BMM category 'settable', namelist
    &settable_ns) multiplied by a fall-out factor found inside the model, held constant
    colder than the fitted range;
  * wall offset dT_BL fitted to the liquid only (refit_score.py --prefix wk3c --constrain);
  * windows start when the hygrometer reaches water saturation (ISKYLAB_TSAT=hygrometer);
  * observed ice: KIT NIce (> 15 um); model ice: crystals 15-218 um by projected-area-
    equivalent diameter (volume-equivalent and maximum dimension as bounds);
  * main comparison: ice at matched gas temperature, before glaciation.
Outputs: paper/figs/fig01-03 (re-drawn), F04-F12 and paper/numbers_v3.json.

Usage: ISKYLAB_TSAT=hygrometer python3 paper_figs_v3.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from netCDF4 import Dataset

import paper_figs as P
import final_numbers as F

assert os.environ.get("ISKYLAB_TSAT") == "hygrometer", "run with ISKYLAB_TSAT=hygrometer"
OUT, FIG = P.OUT, P.FIG
EXPS, MIX, ALONE, DUST = P.EXPS, P.MIX, P.ALONE, P.DUST
INK, MUTED = P.INK, P.MUTED
C0, C1 = "#2a78d6", "#eb6834"
PREFIX = "wk3c"
ACCEPT = 0.3
GRID = np.arange(-21.0, -28.51, -0.5)
FLOOR = 0.02
NAME = {"SDSA01": "South African soil dust (SDSA01)", "ATD03": "Arizona test dust (ATD03)"}
LAB = {"Exp021": "dust alone (Exp021)", "Exp022": "+ NaCl (Exp022)", "Exp023": "+ more NaCl (Exp023)",
       "Exp025": "dust alone (Exp025)", "Exp026": "+ NaCl (Exp026)", "Exp027": "+ more NaCl (Exp027)"}
NS0 = {"SDSA01": (-0.7640, 2.634, -27.75, -20.75), "ATD03": (-1.5437, -13.511, -24.75, -21.75)}
# warm experiments: measured Tw-Tg (Paper 1 v1.7 table) and fitted dT_BL from Paper 1 v1.9 (runs/analysis/v19_settings.txt)
WARM = {"005": (-0.44, -0.28), "006": (-0.25, -0.085), "007": (-0.35, -0.085), "008": (-0.25, -0.09), "009": (-0.25, -0.095), "010": (-0.25, -0.09), "011": (-0.28, -0.09), "013": (-0.27, -0.07), "014": (-0.27, -0.096), "015": (-0.26, -0.07), "016": (-0.26, -0.075), "017": (-1.06, -0.51), "018": (-0.34, -0.075), "019": (-0.26, -0.095)}
tagof = F.tagof


# ------------------------------------------------------------------ helpers
def load_choice():
    ch = F.load_choice()
    rows = list(csv.DictReader(open(OUT / f"refit_scores_{PREFIX}_constrained.csv")))
    acc = {}
    for (e, m) in ch:
        cand = [r for r in rows if r["exp"] == e and int(r["mode"]) == m]
        best = min(float(r["score"]) for r in cand)
        acc[(e, m)] = sorted({r["tag"] for r in cand if float(r["score"]) <= best + ACCEPT})
    return ch, acc, rows


def ns_coef():
    c = {}
    for line in open(OUT / "ns_coef_wk3.txt"):
        if line.strip() and not line.startswith("#"):
            m, d, b, a, tc, cold = line.split()
            c[(int(m), d)] = (float(b), float(a), float(tc))
    return c


_ser = {}


def ice(tag, e, lo=15e-6, hi=218e-6):
    """Model ice (cm-3) within [lo, hi] by area-, volume- and max-dimension size; all ice."""
    key = (tag, e, lo, hi)
    if key in _ser:
        return _ser[key]
    f = P.RUNS / tag / "single_comparison" / f"output-{e}.nc"
    if not f.exists():
        _ser[key] = None
        return None
    v = Dataset(f).variables
    rho = np.asarray(v["p"][:] / (287.05 * v["t"][:]))
    ni, dm, mi = (np.asarray(v[k][:]) for k in ("nicem", "dmaxice", "mice"))
    dv = (6 * mi / (np.pi * 917)) ** (1 / 3)
    da = F.darea_t(dm, np.asarray(v["phi"][:]))
    cnt = lambda d: np.sum(ni * (d > lo) * (d <= hi), axis=(1, 2)) * rho / 1e6
    out = {"t": np.asarray(v["time"][:]), "tg": np.asarray(v["t"][:]) - 273.15,
           "area": cnt(da), "vol": cnt(dv), "dmax": cnt(dm), "all": np.sum(ni, axis=(1, 2)) * rho / 1e6}
    _ser[key] = out
    return out


def nd_mg_obs(o):
    """WELAS droplets per mg of air (cm-3 divided by air density in kg m-3), 31 s smoothed."""
    rho = o["p"] * 100 / (287.05 * (o["tg"] + 273.15))
    return P.smooth(o["nd"], 31) / rho


def nd_mg_mod(m):
    return m["nd"] / m["rhoa"]


def gmean(x):
    x = np.asarray([v for v in x if np.isfinite(v) and v > 0])
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def tglac_model(tag, e):
    m = P.model(tag, e)
    return P.glaciation_T(m["t"], m["tg"], m["lwc"], P.TSAT[e])


def profile(o, e, tag, key="area", lo=15e-6):
    """Observed and modelled ice at matched gas temperature, before glaciation."""
    tgl = F.glac_time(o, e)
    sel = (o["t"] >= P.TSAT[e]) & (o["t"] <= tgl)
    tt, tg, oi = o["t"][sel], o["tg"][sel], P.smooth(o["ni"], 61)[sel]
    s = ice(tag, e, lo=lo)
    lim = max(float(np.interp(tgl, o["t"], o["tg"])), np.nan_to_num(tglac_model(tag, e), nan=-99))
    S = P.initial_psd(e)["S_dust"]
    rows = []
    for T in GRID:
        if T < tg.min() or T > tg.max() or T < lim:
            continue
        k = int(np.argmin(abs(tg - T)))
        rows.append((float(T), float(oi[k]), float(np.interp(tt[k], s["t"], s[key])), float(oi[k] / S)))
    return rows


def om(rows):
    return gmean([ob / mo for T, ob, mo, _ in rows if ob > FLOOR and mo > 0])


def dust_in_drops(tag, e, T):
    """Fraction of the airborne unfrozen dust surface in particles with wet D > 2 um."""
    v = Dataset(P.RUNS / tag / "single_comparison" / f"output-{e}.nc").variables
    tg = np.asarray(v["t"][:]) - 273.15
    i = int(np.argmin(abs(tg - T)))
    n = np.asarray(v["nwat"][i, 0])
    dw = np.asarray(v["dwet"][i, 0])
    md = np.asarray(v["maer"][i, :, 0, :]).sum(0)
    dd = (6 * md / (np.pi * 2600)) ** (1 / 3)          # maer is mass per particle (kg)
    return float(np.sum(n * dd ** 2 * (dw > 2e-6)) / max(np.sum(n * dd ** 2), 1e-30))


def peak_in_drops(tag, e):
    """Largest share of the airborne, unfrozen dust surface in cloud droplets (wet D > 2 um),
    from t_sat to the observed end of the liquid phase."""
    v = Dataset(P.RUNS / tag / "single_comparison" / f"output-{e}.nc").variables
    t = np.asarray(v["time"][:])
    nw, dw = np.asarray(v["nwat"][:, 0]), np.asarray(v["dwet"][:, 0])
    dd2 = (6 * np.asarray(v["maer"][:, :, 0, :]).sum(1) / (np.pi * 2600)) ** (2 / 3)
    f = np.sum(nw * dd2 * (dw > 2e-6), axis=1) / np.maximum(np.sum(nw * dd2, axis=1), 1e-30)
    o = P.obs(e)
    sel = (t >= P.TSAT[e]) & (t <= F.glac_time(o, e))
    return float(f[sel].max())


# ------------------------------------------------------------------ numbers
def numbers(ch, acc, rows):
    O = {e: P.obs(e) for e in EXPS}
    R = {"choice": {f"{e}_m{m}": v for (e, m), v in ch.items()},
         "accept": {f"{e}_m{m}": v for (e, m), v in acc.items()},
         "exp": {}, "liquid": {}, "ice": {}, "matched": {}, "window": {}, "paired": {}, "act": {}, "ns": {}}
    for e in EXPS:
        o = O[e]
        k0 = int(np.argmin(abs(o["t"])))
        R["exp"][e] = {"T0": float(o["tg"][k0]), "p0": float(o["p"][k0]),
                       "RH0w": float(np.nanmean(o["rhw"][o["pre"]])), "RH0i": float(np.nanmean(o["rhi"][o["pre"]])),
                       "tsat": P.TSAT[e], "Tsat": float(np.interp(P.TSAT[e], o["t"], o["tg"])),
                       "dTwg": float(np.nanmean((o["tw"] - o["tg"])[o["t"] >= P.TSAT[e]])),
                       "tend": o["tend"], "Tend": float(np.nanmean(o["tg"][o["t"] > o["tend"] - 30])),
                       "glac_t": F.glac_time(o, e)}
        R["exp"][e]["Tglac"] = float(np.interp(R["exp"][e]["glac_t"], o["t"], o["tg"]))
        for mode in (0, 1):
            tag = tagof(mode, ch[(e, mode)])
            m = P.model(tag, e)
            k = np.where(m["rhw_tot"] >= 100)[0]
            R["exp"][e][f"tsat_m{mode}"] = float(m["t"][k[0]]) if len(k) else float("nan")
            R["exp"][e][f"RH0_m{mode}"] = float(100 * m["rh"][0])
            R["exp"][e][f"Tglac_m{mode}"] = tglac_model(tag, e)
            r = [x for x in rows if x["tag"] == tag and x["exp"] == e][0]
            R["liquid"][f"{e}_m{mode}"] = {k2: float(r[k2]) for k2 in
                                            ("Nd_mod", "Nd_obs", "LWCwelas_mod", "LWC_obs", "TWexcess_mod_pct",
                                             "TWexcess_obs_pct", "score")}
    # matched temperature (main comparison)
    for key in ("area", "vol", "dmax", "area20"):
        k2, lo = ("area", 20e-6) if key == "area20" else (key, 15e-6)
        prof = {e: profile(O[e], e, tagof(1, ch[(e, 1)]), k2, lo) for e in EXPS}
        d = {"profiles": prof, "OM": {e: om(prof[e]) for e in EXPS}, "E": {}, "Eobs": {}}
        for e in MIX:
            a = ALONE[e]
            d["E"][e] = d["OM"][e] / d["OM"][a]
            pa = {T: n for T, ob, mo, n in prof[a] if ob > FLOOR}
            d["Eobs"][e] = gmean([n / pa[T] for T, ob, mo, n in prof[e] if ob > FLOOR and T in pa])
            d.setdefault("raw", {})[e] = gmean([ob / {T2: o2 for T2, o2, _, _ in prof[a]}[T]
                                                 for T, ob, mo, n in prof[e] if ob > FLOOR and T in pa])
            d.setdefault("Trange", {})[e] = [max(T for T, ob, *_ in prof[e] if ob > FLOOR and T in pa),
                                              min(T for T, ob, *_ in prof[e] if ob > FLOOR and T in pa),
                                              sum(1 for T, ob, *_ in prof[e] if ob > FLOOR and T in pa)]
        R["matched"][key] = d
    # E over all acceptable fits (mode 1, area)
    R["matched"]["E_range"] = {}
    for e in MIX:
        a = ALONE[e]
        vals = []
        for tm in acc[(e, 1)]:
            for ta in acc[(a, 1)]:
                x, y = om(profile(O[e], e, tm)), om(profile(O[a], a, ta))
                vals.append(x / y)
        R["matched"]["E_range"][e] = [min(vals), max(vals), len(vals)]
    # mode 0 at matched temperature: mean ice over the matched range
    R["matched"]["mode0"] = {}
    for e in EXPS:
        pr = profile(O[e], e, tagof(0, ch[(e, 0)]))
        pr1 = R["matched"]["area"]["profiles"][e]
        Ts = {T for T, ob, *_ in pr1 if ob > FLOOR}
        R["matched"]["mode0"][e] = {"obs": float(np.mean([ob for T, ob, mo, _ in pr1 if T in Ts])),
                                    "m1": float(np.mean([mo for T, ob, mo, _ in pr1 if T in Ts])),
                                    "m0": float(np.mean([np.interp(T, [x[0] for x in pr][::-1], [x[2] for x in pr][::-1])
                                                         for T in Ts])) if pr else 0.0}
    # window means (secondary): pre-glaciation and full window
    for e in EXPS:
        o = O[e]
        pre, full, _ = F.windows(o, e)
        d = {"obs_pre": float(np.nanmean(P.smooth(o["ni"], 61)[pre])), "obs_full": float(np.nanmean(o["ni"][full]))}
        for mode in (0, 1):
            s = ice(tagof(mode, ch[(e, mode)]), e)
            d[f"m{mode}_pre"] = F.mean_at(s, "area", o["t"][pre])
            d[f"m{mode}_full"] = F.mean_at(s, "area", o["t"][full])
        R["window"][e] = d
    for e in MIX:
        a = ALONE[e]
        for w in ("pre", "full"):
            R["window"][e][f"R_obs_{w}"] = R["window"][e][f"obs_{w}"] / R["window"][a][f"obs_{w}"]
            for mode in (0, 1):
                den = R["window"][a][f"m{mode}_{w}"]
                R["window"][e][f"R_m{mode}_{w}"] = R["window"][e][f"m{mode}_{w}"] / den if den > 0 else float("nan")
    # paired simulations (NaCl removed)
    for e in MIX:
        o = O[e]
        pre, _, _ = F.windows(o, e)
        for mode in (0, 1):
            tag = tagof(mode, ch[(e, mode)])
            mix, dus = ice(tag, e), ice(tag + "dust", e)
            if dus is None:
                continue
            p = {}
            for k in ("area", "all"):
                den = F.mean_at(dus, k, o["t"][pre])
                p[k] = F.mean_at(mix, k, o["t"][pre]) / den if den > 0 else float("nan")
            mm, md = P.model(tag, e), P.model(tag + "dust", e)
            p.update(ret_mix=float(mm["dust_ret"][-1]), ret_dust=float(md["dust_ret"][-1]),
                     Tglac_mix=tglac_model(tag, e), Tglac_dust=tglac_model(tag + "dust", e))
            R["paired"][f"{e}_m{mode}"] = p
    # where the dust is (model, mode 1)
    for e in EXPS:
        tag = tagof(1, ch[(e, 1)])
        R["act"][e] = {f"{T}": dust_in_drops(tag, e, T) for T in (-21, -23)}
        if e in MIX:
            R["act"][e + "dust"] = {f"{T}": dust_in_drops(tag + "dust", e, T) for T in (-21, -23)}
    # n_s calibration factors
    c = ns_coef()
    for (mode, d), (b, a, tc) in c.items():
        R["ns"][f"{d}_m{mode}"] = {"b": b, "a": a, "tc": tc, "F": float(np.exp(a - NS0[d][1]))}
    return R, O


# ------------------------------------------------------------------ figures
def style(ax, letter=None, loc="left"):
    ax.grid(alpha=0.25, lw=0.4)
    if letter:
        x, ha = (0.02, "left") if loc == "left" else (0.98, "right")
        ax.text(x, 0.97, letter, transform=ax.transAxes, ha=ha, va="top", fontweight="bold")


def tx(ax):
    ax.set_xlim(-15.5, -31)


def fig_onset(R, O):
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.0), sharey=True)
    for ax, e in zip(axs.flat, EXPS):
        o = O[e]
        c = (o["t"] > -30) & (o["t"] < P.TSAT[e] + 240)
        ax.plot(o["t"][c] / 60, P.smooth(o["rhw"], 31)[c], color=INK, lw=1.8, label="observed (hygrometer)")
        for k, ls, lw, col in (("01", ":", 1.0, MUTED), ("03", "-", 1.5, C1), ("08", "--", 1.0, MUTED)):
            m = P.model(f"wk_m1_k{k}", e)
            if m is None:
                continue
            cc = m["t"] < P.TSAT[e] + 240
            ax.plot(m["t"][cc] / 60, m["rhw_tot"][cc], ls, color=col, lw=lw,
                    label=f"model, k$_m$ = {int(k)}×10$^{{-4}}$ m s$^{{-1}}$")
        ax.axhline(100, color=MUTED, lw=0.5)
        ax.axvline(P.TSAT[e] / 60, color=INK, lw=0.6, ls="--")
        ax.set_title(f"{e} ({'SDSA01' if DUST[e] == 'SDSA01' else 'ATD03'}{' + NaCl' if e in MIX else ''})",
                     loc="left", fontsize=7.5)
        ax.set_ylim(66, 104)
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axs[1]:
        ax.set_xlabel("Time since the start of pumping (min)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Total water, RH$_w$ (%)")
    axs[0, 0].legend(fontsize=5.8, loc="lower right")
    fig.tight_layout()
    P.save(fig, "F04_wall_sink")


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
        dx = R["exp"][e]["dTwg"]
        ax.plot(dx, R["choice"][f"{e}_m1"], "D", ms=5, color=C1, zorder=4)
        ax.annotate(e[3:], (dx, R["choice"][f"{e}_m1"]), textcoords="offset points", xytext={"Exp022": (-17, 5)}.get(e, (5, -2)), fontsize=6)
    ax.plot([], [], "o", color=INK, ms=4, label="warm (Paper 1, v1.9)")
    ax.plot([], [], "s", color=INK, mfc="white", ms=4, label="warm, Exp005 and Exp017")
    ax.plot([], [], "D", color=C1, ms=5, label="cold, this study (mode 1)")
    ax.set_xlabel("Measured T$_{wall}$ − T$_{gas}$ (K)")
    ax.set_ylabel("Fitted ΔT$_{BL}$ (K)")
    ax.set_xlim(-1.15, 0.55)
    ax.set_ylim(-0.6, 0.6)
    ax.legend(fontsize=5.8, loc="upper left")
    style(ax)
    fig.tight_layout()
    P.save(fig, "F05_wall_offsets")


def fig_ns(R, ch):
    ns = {}
    for r in csv.DictReader(open(P.HERE / "figs" / "ns_observed.csv")):
        if r["dilution"] == "cpc":
            ns.setdefault(r["exp"], {})[float(r["T_mid_C"])] = P._num(r["ns_dust_m2"])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    pub = {"SDSA01": (lambda t: np.exp(-0.517 * t + 8.934), (-36, -12), "Niemand et al. (2012), desert dust")}
    for ax, (e, d) in zip(axs, (("Exp021", "SDSA01"), ("Exp025", "ATD03"))):
        b, a, tc, tw = NS0[d]
        T = np.array(sorted(ns[e]))
        y = np.array([ns[e][t] for t in T])
        ok = np.isfinite(y) & (T >= -30.5) & (y > 0)
        ax.semilogy(T[ok], y[ok], "o", ms=3.2, color=INK, label=f"observed, airborne ice ({e})")
        # corrected for fall-out with the model's airborne fraction (mode 1 run)
        v = Dataset(P.RUNS / tagof(1, ch[(e, 1)]) / "single_comparison" / f"output-{e}.nc").variables
        rho = np.asarray(v["p"][:] / (287.05 * v["t"][:]))
        tg = np.asarray(v["t"][:]) - 273.15
        air, fall = np.asarray(v["nice"][:]) * rho, np.asarray(v["nfall_ice"][:]) * rho
        Tc, yc = [], []
        for t in T[ok]:
            if t > -18:
                continue
            k = np.argmin(abs(tg - t))
            f = air[k] / max(air[k] + fall[k], 1e-30)
            if 0.02 < f <= 1:
                Tc.append(t)
                yc.append(ns[e][t] / f)
        ax.semilogy(Tc, yc, "o", ms=3.2, color=C1, mfc="white", label="observed ÷ airborne fraction (model)")
        tt = np.linspace(-18, -31, 200)
        ax.semilogy(tt, np.where(tt > -18, np.nan, np.exp(b * np.maximum(tt, tc) + a)), "--", color=INK, lw=1.0,
                    label="fit to the airborne ice")
        c = R["ns"][f"{d}_m1"]
        ax.semilogy(tt, np.exp(c["b"] * np.maximum(tt, c["tc"]) + c["a"]), "-", color=C1, lw=1.8,
                    label=f"used in the model (× {c['F']:.1f}), constant colder than {c['tc']:.1f} °C")
        if d in pub:
            f, (lo, hi), lab = pub[d]
            t2 = np.linspace(lo, hi, 50)
            ax.semilogy(t2, f(t2), ":", color=MUTED, lw=1.2, label=lab)
        ax.axvspan(tc, tw, color=C1, alpha=0.08, lw=0)
        ax.set_ylim(1e6, 3e11)
        ax.set_xlim(-17.5, -31)
        ax.set_xlabel("Gas temperature (°C)")
        ax.set_title(NAME[d], loc="left")
        ax.legend(fontsize=5.8, loc="lower right")
        style(ax, "ab"[0 if d == "SDSA01" else 1])
    axs[0].set_ylabel("n$_s$ (m$^{-2}$)")
    fig.tight_layout()
    P.save(fig, "F06_ns_calibration")


def fig_liquid_alone(R, O, ch):
    fig, axs = plt.subplots(3, 2, figsize=(7.0, 6.0), sharex=True)
    for j, e in enumerate(("Exp021", "Exp025")):
        o = O[e]
        c = o["t"] > 0
        ndo = nd_mg_obs(o)
        pre = np.nanmean(ndo[o["pre"]])
        axs[0, j].plot(o["tg"][c], ndo[c] - pre, color=INK, label="observed")
        axs[1, j].plot(o["tg"][c], P.smooth(o["lwc"], 31)[c], color=INK)
        axs[2, j].plot(o["tg"][c], P.smooth(o["rhw"], 31)[c] - 100, color=INK)
        for mode, col in ((0, C0), (1, C1)):
            m = P.model(tagof(mode, ch[(e, mode)]), e)
            ndm = nd_mg_mod(m)
            axs[0, j].plot(m["tg"], ndm - ndm[0], color=col, label=f"model, mode {mode}")
            axs[1, j].plot(m["tg"], m["lwc_welas"], color=col)
            axs[2, j].plot(m["tg"], m["rhw_tot"] - 100, color=col)
        axs[0, j].set_title(f"{e}: {DUST[e]} alone, ΔT$_{{BL}}$ = {ch[(e, 1)]:+.2f} K", loc="left")
        axs[2, j].set_xlabel("Gas temperature (°C)")
        for i in range(3):
            tx(axs[i, j])
            style(axs[i, j], "abcdef"[2 * i + j], "right")
            axs[i, j].axvline(R["exp"][e]["Tsat"], color=MUTED, lw=0.6, ls="--")
        axs[2, j].axhline(0, color=MUTED, lw=0.6)
        axs[0, j].axhline(0, color=MUTED, lw=0.6)
    axs[0, 0].set_ylabel("Droplets above pre-cloud\n(per mg of air)")
    axs[1, 0].set_ylabel("Liquid water seen\nby WELAS (g m$^{-3}$)")
    axs[2, 0].set_ylabel("Total water above\nwater saturation (%)")
    axs[2, 0].legend(*axs[0, 0].get_legend_handles_labels(), fontsize=6.5, loc="lower left")
    fig.tight_layout()
    P.save(fig, "F07_liquid_dust_alone")


def fig_liquid_mix(R, O, ch):
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.0), sharex=True, sharey="row")
    for j, e in enumerate(MIX):
        o = O[e]
        c = o["t"] > 0
        ndo = nd_mg_obs(o)
        pre = np.nanmean(ndo[o["pre"]])
        axs[0, j].plot(o["tg"][c], P.smooth(o["rhw"], 31)[c] - 100, color=INK, lw=1.6, label="observed")
        axs[1, j].plot(o["tg"][c], ndo[c] - pre, color=INK, lw=1.6)
        for mode, col, ls, lw in ((0, C0, "--", 1.0), (1, C1, "-", 1.4)):
            m = P.model(tagof(mode, ch[(e, mode)]), e)
            tw = m["rhw_tot"] - 100
            axs[0, j].plot(m["tg"], np.where(tw > 19, np.nan, tw), ls, color=col, lw=lw, label=f"model, mode {mode}")
            if mode == 0 and np.nanmax(tw) > 19:
                k = int(np.argmax(tw > 19))
                axs[0, j].annotate(f"→ +{np.nanmax(tw):.0f} %", (m["tg"][k], 19), fontsize=5.8, color=col,
                                   textcoords="offset points", xytext=(2, -8))
            ndm = nd_mg_mod(m)
            y = P.smooth(ndm - ndm[0], 3)
            axs[1, j].plot(m["tg"], np.where(y > 290, np.nan, y), ls, color=col, lw=lw)
            if np.nanmax(y) > 290:
                k = int(np.argmax(y > 290))
                axs[1, j].annotate(f"↑ {np.nanmax(y):.0f}", (m["tg"][k], 285), fontsize=5.8, color=col,
                                   textcoords="offset points", xytext=(3 if mode else -16, -2))
        axs[0, j].set_title(f"{e} ({DUST[e]} + NaCl)", loc="left", fontsize=7.2)
        axs[1, j].set_xlabel("Gas T (°C)")
        for i in range(2):
            tx(axs[i, j])
            style(axs[i, j], "abcdefgh"[4 * i + j], "right")
            axs[i, j].axhline(0, color=MUTED, lw=0.6)
            axs[i, j].axvline(R["exp"][e]["Tglac"], color=INK, lw=0.6, ls=":")
    axs[0, 0].set_ylabel("Total water above\nwater saturation (%)")
    axs[1, 0].set_ylabel("Droplets above pre-cloud\n(per mg of air)")
    axs[0, 0].set_ylim(-22, 21)
    axs[1, 0].set_ylim(-60, 300)
    axs[0, 0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    P.save(fig, "F08_liquid_mixtures")


def fig_ice(R, O, ch, acc):
    fig, axs = plt.subplots(3, 2, figsize=(7.0, 6.6), sharex=True, sharey=True)
    order = (("Exp021", "Exp025"), ("Exp022", "Exp026"), ("Exp023", "Exp027"))
    for i, row in enumerate(order):
        for j, e in enumerate(row):
            ax = axs[i, j]
            o = O[e]
            c = o["t"] > 0
            ax.semilogy(o["tg"][c], P.smooth(o["ni"], 61)[c], color=INK, lw=1.6, label="observed (WELAS, > 15 µm)")
            for mode, col in ((0, C0), (1, C1)):
                s = ice(tagof(mode, ch[(e, mode)]), e)
                ax.semilogy(s["tg"], np.clip(s["area"], 1e-4, None), color=col,
                            label=f"model, mode {mode} (15–218 µm)")
                runs = [ice(t, e) for t in acc[(e, mode)]]
                if len(runs) > 1:
                    arr = np.array([np.interp(s["t"], r["t"], r["area"]) for r in runs])
                    ax.fill_between(s["tg"], np.clip(arr.min(0), 1e-4, None), np.clip(arr.max(0), 1e-4, None),
                                    color=col, alpha=0.15, lw=0)
                if mode == 1:
                    ax.semilogy(s["tg"], np.clip(s["vol"], 1e-4, None), ":", color=col, lw=0.9)
            ax.axvline(R["exp"][e]["Tglac"], color=INK, lw=0.6, ls="--")
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
    P.save(fig, "F09_ice")


def fig_matched(R):
    A = R["matched"]["area"]
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True)
    for j, (d, exps) in enumerate((("SDSA01", ["Exp021", "Exp022", "Exp023"]), ("ATD03", ["Exp025", "Exp026", "Exp027"]))):
        for e in exps:
            pr = np.array(A["profiles"][e])
            ok = pr[:, 1] > FLOOR
            c = P.C_RUN[e]
            lw = 2.0 if e not in MIX else 1.4
            axs[0, j].plot(pr[ok, 0], pr[ok, 3] * 1e2, "-", color=c, lw=lw, marker="o", ms=2.5, label=LAB[e])
            axs[1, j].plot(pr[ok, 0], pr[ok, 1] / pr[ok, 2], "-", color=c, lw=lw, marker="o", ms=2.5)
            if e in MIX:
                k = exps.index(e)
                lo, hi, _ = R["matched"]["E_range"][e]
                axs[1, j].text(0.03, 0.97 - 0.07 * k,
                               f"{LAB[e].split(' (')[0]}: ×{A['E'][e]:.1f} ({lo:.1f}–{hi:.1f}) relative to dust alone",
                               transform=axs[1, j].transAxes, va="top", fontsize=6.3, color=c)
        axs[0, j].set_title(NAME[d], fontsize=8, loc="left")
        axs[0, j].set_yscale("log")
        axs[1, j].set_yscale("log")
        axs[1, j].axhline(1, color=MUTED, lw=0.6, zorder=0)
        axs[1, j].set_xlabel("Gas temperature (°C)")
        axs[1, j].set_ylim(0.1, 30)
        axs[0, j].legend(fontsize=6.5, loc="upper left" if j == 0 else "lower right")
    for ax, l in zip(axs.flat, "abcd"):
        ax.set_xlim(-20.5, -26.5)
        ax.grid(alpha=0.25, lw=0.4)
        ax.text(0.99, 0.97, l, transform=ax.transAxes, ha="right", va="top", fontweight="bold")
    axs[0, 0].set_ylabel("Observed ice per dust surface\n(cm$^{-3}$ per 100 µm$^2$ cm$^{-3}$)")
    axs[1, 0].set_ylabel("Observed ÷ modelled ice\n(model: mode 1, no NaCl effect)")
    fig.tight_layout()
    P.save(fig, "F10_matched_T")


def fig_mech(R, ch):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.25, 1]})
    ax = axs[0]
    x = np.arange(len(EXPS))
    pk = {e: peak_in_drops(tagof(1, ch[(e, 1)]), e) for e in EXPS}
    pkd = {e: peak_in_drops(tagof(1, ch[(e, 1)]) + "dust", e) for e in MIX}
    R["act_peak"] = {**pk, **{e + "dust": v for e, v in pkd.items()}}
    ax.bar(x, [pk[e] for e in EXPS], width=0.55, color=[INK if e not in MIX else C1 for e in EXPS], label="as run")
    xm = [EXPS.index(e) for e in MIX]
    ax.plot(xm, [pkd[e] for e in MIX], "o", mfc="white", color=MUTED, ms=5, label="NaCl removed")
    for e in MIX:
        ax.text(EXPS.index(e), 0.02, "0", ha="center", fontsize=6, color=C1)
    ax.set_xticks(x)
    ax.set_xticklabels([e[3:] for e in EXPS])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Largest share of the dust\nsurface in cloud droplets")
    ax.set_title("(a) Model: where the dust is", loc="left", fontsize=7.5)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(0.0, 0.68))
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    ax = axs[1]
    cols = {"Exp022": "#1baf7a", "Exp023": "#e87ba4", "Exp026": "#2a78d6", "Exp027": "#eb6834"}
    for e in MIX:
        tag = tagof(1, ch[(e, 1)])
        mix, dus = P.model(tag, e), P.model(tag + "dust", e)
        ax.plot(mix["tg"], mix["dust_ret"], "-", color=cols[e], label=e)
        ax.plot(dus["tg"], dus["dust_ret"], ":", color=cols[e])
    ax.plot([], [], "-", color=MUTED, label="with NaCl")
    ax.plot([], [], ":", color=MUTED, label="NaCl removed")
    tx(ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Gas temperature (°C)")
    ax.set_ylabel("Fraction of dust still airborne")
    ax.set_title("(b) Model, mode 1", loc="left", fontsize=7.5)
    ax.legend(fontsize=5.8, loc="lower left", ncol=2)
    ax.grid(alpha=0.25, lw=0.4)
    ax = axs[2]
    ax.plot(x - 0.15, [R["exp"][e]["Tglac"] for e in EXPS], "o", color=INK, ms=5, label="observed")
    ax.plot(x, [R["exp"][e]["Tglac_m1"] for e in EXPS], "s", color=C1, ms=5, label="model, mode 1")
    ax.plot(x + 0.15, [R["exp"][e]["Tglac_m0"] for e in EXPS], "s", color=C0, ms=4, label="model, mode 0")
    ax.set_xticks(x)
    ax.set_xticklabels([e[3:] for e in EXPS])
    ax.set_xlabel("Experiment")
    ax.set_ylabel("End of the liquid phase (°C)")
    ax.set_ylim(-23.5, -30.5)
    ax.set_title("(c) Glaciation", loc="left", fontsize=7.5)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    P.save(fig, "F11_mechanisms")


def fig_schematic(R):
    A = R["matched"]["area"]["E"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0))
    titles = ("(a) Dust alone: the dust activates\nand freezes in droplets",
              "(b) + NaCl, if dust must activate\n(mode 0): almost no ice",
              "(c) + NaCl, observed: the unactivated\ndust froze")
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
    ax.text(5.0, 4.6, f"model: up to {100 * R['act_peak']['Exp021']:.0f} % (SDSA01) and\n"
                      f"{100 * R['act_peak']['Exp025']:.0f} % (ATD03) of the dust\nsurface in droplets",
            fontsize=6, ha="center")
    ax = axs[1]
    ax.text(0.2, 9.0, "NaCl takes the vapour", fontsize=6.3)
    for (x, y) in ((1.0, 7.3), (2.4, 6.4), (3.4, 7.8), (6.6, 7.6), (8.2, 6.6), (7.6, 8.4), (1.6, 4.6), (8.8, 4.8)):
        drop(ax, x, y, 0.42, c=salt_c, a=0.6)
    dust(ax, 5.0, 6.6)
    ax.text(5.0, 5.4, "dust does not activate", fontsize=6, ha="center")
    ax.text(5.0, 4.5, "→ cannot freeze", fontsize=6, ha="center")
    m0 = R["matched"]["mode0"]
    hi = max(m0[e]["m0"] for e in MIX)
    lo_o = min(m0[e]["obs"] for e in MIX)
    ax.text(5.0, 2.2, f"model: ≤ {hi:.2f} cm$^{{-3}}$ in the mixtures\nobserved: ≥ {lo_o:.2f} cm$^{{-3}}$",
            fontsize=6.3, ha="center", color=C0)
    ax = axs[2]
    ax.text(0.2, 9.0, "water adsorbed on the dust", fontsize=6.3)
    for (x, y) in ((1.0, 7.3), (2.6, 8.0), (7.0, 8.2), (8.6, 6.8), (1.4, 4.6), (8.8, 4.6)):
        drop(ax, x, y, 0.42, c=salt_c, a=0.6)
    drop(ax, 4.0, 6.6, 0.38)
    dust(ax, 4.0, 6.6)
    arrow(ax, 4.6, 6.6, 6.2, 6.6, "freezes")
    crystal(ax, 7.1, 6.0)
    dust(ax, 7.1, 6.0, 0.18)
    ax.text(5.0, 3.9, "ice per unit dust surface,\nrelative to the dust alone:", fontsize=6, ha="center")
    ax.text(5.0, 2.0, f"ATD03: × {A['Exp026']:.1f} and {A['Exp027']:.1f}\nSDSA01: × {A['Exp022']:.1f} and {A['Exp023']:.1f}",
            fontsize=6.3, ha="center", color=C1)
    fig.tight_layout()
    P.save(fig, "F12_schematic")


def obs_figs(O):
    """Figures 1-3 from paper_figs.py (observations only)."""
    ns = {}
    for r in csv.DictReader(open(P.HERE / "figs" / "ns_observed.csv")):
        if r["dilution"] == "cpc":
            ns.setdefault(r["exp"], {})[float(r["T_mid_C"])] = P._num(r["ns_dust_m2"])
    P.fig_aerosol(None)
    P.fig_overview(None, O, ns)
    P.fig_psd(O)


def main():
    ch, acc, rows = load_choice()
    R, O = numbers(ch, acc, rows)
    with open(OUT / "numbers_v3.json", "w") as fh:
        json.dump(R, fh, indent=1, default=float)
    obs_figs(O)
    fig_onset(R, O)
    fig_wall(R)
    fig_ns(R, ch)
    fig_liquid_alone(R, O, ch)
    fig_liquid_mix(R, O, ch)
    fig_ice(R, O, ch, acc)
    fig_matched(R)
    fig_mech(R, ch)
    fig_schematic(R)
    print("written", OUT / "numbers_v3.json", "and figures in", FIG)


if __name__ == "__main__":
    main()
