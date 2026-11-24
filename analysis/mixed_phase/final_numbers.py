"""Final comparison for the mixed-phase paper (17 Nov 2026), runs wk3c_*:
measured start humidity, aerosol at t=0, wall vapour sink km = 3e-4 m/s, n_s fitted to
the dust-alone runs and corrected for fall-out (ns_calibrate.py), wall offset fitted to
the liquid only within the gas-wall range (refit_score.py --prefix wk3c --constrain).

Ice operator (model): crystals whose WELAS-like size lies in 15-218 um.  WELAS gives a
scattering-equivalent diameter; for crystals much larger than the wavelength this
follows projected area, so the main operator uses the projected-area-equivalent
diameter of a randomly oriented spheroid (maximum dimension and aspect from BMM).
Volume-equivalent and maximum-dimension sizes are given as bounds.

Ice metric: mean ice from water saturation (hygrometer) to the observed end of the
liquid phase ("pre-glaciation"), the same times for observations and model.  The
full-window mean (saturation + 5 min to the end) is also reported.

Usage: ISKYLAB_TSAT=hygrometer python3 final_numbers.py   -> paper/numbers_v3.json
"""
import json
import os

import numpy as np
from netCDF4 import Dataset

import paper_figs as P

assert os.environ.get("ISKYLAB_TSAT") == "hygrometer", "run with ISKYLAB_TSAT=hygrometer"
PREFIX = "wk3c"
LO, HI = 15e-6, 218e-6   # KIT NIce is > 15 um (Datasets-V2 header)
OUT = P.OUT


def tagof(mode, off):
    v = round(off * 1000)
    return f"{PREFIX}_m{mode}_{'m' if v < 0 else 'p'}{abs(v):03d}"


def darea(dmax, phi):
    """Projected-area-equivalent diameter of a randomly oriented spheroid with maximum
    dimension dmax and aspect phi (>1 prolate column, <1 oblate plate).  The mean
    projected area of a convex body is S/4, so D_area = sqrt(S/pi)."""
    phi = np.where((phi > 0) & np.isfinite(phi), phi, 1.0)
    phi = np.where(np.abs(phi - 1) < 1e-4, 1.0001, phi)
    if np.all(phi > 1):                      # prolate: dmax = 2c, a = c/phi
        c = dmax / 2
        a = c / phi
        e = np.sqrt(1 - 1 / phi ** 2)
        S = 2 * np.pi * a ** 2 * (1 + (c / (a * e)) * np.arcsin(e))
    else:                                    # oblate: dmax = 2a, c = a*phi
        a = dmax / 2
        e = np.sqrt(1 - phi ** 2)
        S = 2 * np.pi * a ** 2 * (1 + ((1 - e ** 2) / e) * np.arctanh(e))
    return np.sqrt(S / np.pi)


def darea_t(dmax, phi_t):
    out = np.empty_like(dmax)
    for i, ph in enumerate(phi_t):
        out[i] = darea(dmax[i], np.full(dmax[i].shape, ph))
    return out


_ice = {}


def ice_series(tag, e):
    """Model ice (cm-3) in 15-218 um by area-, volume- and max-dimension size; all ice."""
    key = (tag, e)
    if key in _ice:
        return _ice[key]
    f = P.RUNS / tag / "single_comparison" / f"output-{e}.nc"
    if not f.exists():
        _ice[key] = None
        return None
    v = Dataset(f).variables
    rho = np.asarray(v["p"][:] / (287.05 * v["t"][:]))
    ni = np.asarray(v["nicem"][:])
    dm = np.asarray(v["dmaxice"][:])
    mi = np.asarray(v["mice"][:])
    phi = np.asarray(v["phi"][:])
    dv = (6 * mi / (np.pi * 917)) ** (1 / 3)
    da = darea_t(dm, phi)
    cnt = lambda d: np.sum(ni * (d > LO) * (d <= HI), axis=(1, 2)) * rho / 1e6
    out = {"t": np.asarray(v["time"][:]), "tg": np.asarray(v["t"][:]) - 273.15,
           "area": cnt(da), "vol": cnt(dv), "dmax": cnt(dm), "all": np.sum(ni, axis=(1, 2)) * rho / 1e6}
    _ice[key] = out
    return out


def load_choice():
    ch = {}
    for line in open(OUT / f"refit_choice_{PREFIX}_constrained.txt"):
        if line.strip() and not line.startswith("#"):
            e, m, off = line.split()
            ch[(e, int(m))] = float(off)
    return ch


def windows(o, e):
    ts = P.TSAT[e]
    tgl = glac_time(o, e)
    pre = (o["t"] >= ts) & (o["t"] <= tgl)
    full = (o["t"] >= ts + 300) & (o["t"] <= o["tend"])
    return pre, full, tgl


def glac_time(o, e):
    s = P.smooth(o["lwc"], 61)
    after = o["t"] >= P.TSAT[e] - 120
    imax = int(np.nanargmax(np.where(after, s, -np.inf)))
    later = np.where((np.arange(len(s)) > imax) & (s < 0.1 * s[imax]))[0]
    return float(o["t"][later[0]]) if len(later) else float(o["tend"])


def mean_at(series, key, tt):
    return float(np.nanmean(np.interp(tt, series["t"], series[key])))


def main():
    ch = load_choice()
    O = {e: P.obs(e) for e in P.EXPS}
    R = {"choice": {f"{e}_m{m}": v for (e, m), v in ch.items()}, "obs": {}, "model": {}, "ratio": {}, "paired": {}}
    for e in P.EXPS:
        o = O[e]
        pre, full, tgl = windows(o, e)
        oi = P.smooth(o["ni"], 61)
        R["obs"][e] = {"t_sat": P.TSAT[e], "t_glac": tgl, "T_glac": float(np.interp(tgl, o["t"], o["tg"])),
                       "Ni_pre": float(np.nanmean(oi[pre])), "Ni_full": float(np.nanmean(o["ni"][full])),
                       "Ni_peak_pre": float(np.nanmax(oi[pre]))}
        for mode in (0, 1):
            tag = tagof(mode, ch[(e, mode)])
            s = ice_series(tag, e)
            d = {"tag": tag, "dTBL": ch[(e, mode)]}
            for k in ("area", "vol", "dmax", "all"):
                d[f"Ni_pre_{k}"] = mean_at(s, k, o["t"][pre])
                d[f"Ni_full_{k}"] = mean_at(s, k, o["t"][full])
            m = P.model(tag, e)
            d["T_glac"] = P.glaciation_T(m["t"], m["tg"], m["lwc"], P.TSAT[e])
            R["model"][f"{e}_m{mode}"] = d
    for e in P.MIX:
        a = P.ALONE[e]
        r = {}
        for w in ("pre", "full"):
            r[f"obs_{w}"] = R["obs"][e][f"Ni_{w}"] / R["obs"][a][f"Ni_{w}"]
            for mode in (0, 1):
                for k in ("area", "vol", "dmax", "all"):
                    num = R["model"][f"{e}_m{mode}"][f"Ni_{w}_{k}"]
                    den = R["model"][f"{a}_m{mode}"][f"Ni_{w}_{k}"]
                    r[f"m{mode}_{w}_{k}"] = num / den if den > 0 else float("nan")
        surf = P.initial_psd(a)["S_dust"] / P.initial_psd(e)["S_dust"]
        r["obs_pre_per_surface"] = r["obs_pre"] * surf
        r["obs_full_per_surface"] = r["obs_full"] * surf
        R["ratio"][e] = r
        o = O[e]
        pre, full, _ = windows(o, e)
        for mode in (0, 1):
            tag = tagof(mode, ch[(e, mode)])
            mix, dus = ice_series(tag, e), ice_series(tag + "dust", e)
            if dus is None:
                continue
            p = {}
            for w, sel in (("pre", pre), ("full", full)):
                for k in ("area", "all"):
                    den = mean_at(dus, k, o["t"][sel])
                    p[f"{w}_{k}"] = mean_at(mix, k, o["t"][sel]) / den if den > 0 else float("nan")
            mm, md = P.model(tag, e), P.model(tag + "dust", e)
            p["ret_mix"], p["ret_dust"] = float(mm["dust_ret"][-1]), float(md["dust_ret"][-1])
            p["Tglac_mix"] = P.glaciation_T(mm["t"], mm["tg"], mm["lwc"], P.TSAT[e])
            p["Tglac_dust"] = P.glaciation_T(md["t"], md["tg"], md["lwc"], P.TSAT[e])
            R["paired"][f"{e}_m{mode}"] = p
    with open(OUT / "numbers_v3.json", "w") as fh:
        json.dump(R, fh, indent=1, default=float)

    f = lambda x: f"{x:.3g}"
    print("Offsets:", R["choice"])
    print("\nIce, pre-glaciation mean (cm-3): obs | mode 0 area (vol, dmax) | mode 1 area (vol, dmax)   [obs glaciation T]")
    for e in P.EXPS:
        m0, m1 = R["model"][f"{e}_m0"], R["model"][f"{e}_m1"]
        print(f"  {e}: {f(R['obs'][e]['Ni_pre'])} | {f(m0['Ni_pre_area'])} ({f(m0['Ni_pre_vol'])}, {f(m0['Ni_pre_dmax'])}) | "
              f"{f(m1['Ni_pre_area'])} ({f(m1['Ni_pre_vol'])}, {f(m1['Ni_pre_dmax'])})   [{R['obs'][e]['T_glac']:.1f} C; model m1 {m1['T_glac']:.1f}]")
    print("\nRatio mixture / dust alone, pre-glaciation: obs (per surface) | mode 0 area | mode 1 area (vol, dmax)")
    for e in P.MIX:
        r = R["ratio"][e]
        print(f"  {e}: {f(r['obs_pre'])} ({f(r['obs_pre_per_surface'])}) | {f(r['m0_pre_area'])} | "
              f"{f(r['m1_pre_area'])} ({f(r['m1_pre_vol'])}, {f(r['m1_pre_dmax'])})")
    print("\nRatio, full window: obs | mode 0 area | mode 1 area")
    for e in P.MIX:
        r = R["ratio"][e]
        print(f"  {e}: {f(r['obs_full'])} | {f(r['m0_full_area'])} | {f(r['m1_full_area'])}")
    print("\nPaired (NaCl removed), pre-glaciation: area-sized | all crystals; dust airborne at end mix/no-NaCl")
    for k, p in R["paired"].items():
        print(f"  {k}: {f(p['pre_area'])} | {f(p['pre_all'])}; {p['ret_mix']:.2f}/{p['ret_dust']:.2f}; glaciation {p['Tglac_mix']:.1f}/{p['Tglac_dust']:.1f}")


if __name__ == "__main__":
    main()
