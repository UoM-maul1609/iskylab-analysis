"""Ice at matched gas temperature (17 Nov 2026).

Under a singular (time-independent) description, ice depends on the dust surface and the
coldest temperature reached, so experiments are compared at the same gas temperature,
not over windows that end at different temperatures.

For each experiment, on a 0.5 K grid from -21 C to the warmer of the observed and modelled
(mode 1) glaciation temperatures, with observed ice > 0.02 cm-3:
  O/M  = observed ice (WELAS NIce, > 15 um, 61 s smoothed) / modelled ice (mode 1;
         crystals 15-218 um by projected-area-equivalent size).  Mode 1 freezes dust
         without activation, so it carries the dust surface, losses, temperature and
         cooling history of each run but no suppression by NaCl.
  E    = (O/M)_mixture / (O/M)_dust alone: the NaCl effect the model does not contain.
  Eobs = [Ni/S_dust]_mixture / [Ni/S_dust]_dust alone, observations only.
Geometric means over the grid.  Bounds: volume-equivalent and maximum-dimension sizes.
Writes paper/matched_T.json and prints a table.
"""
import json
import os

import numpy as np

import paper_figs as P
import final_numbers as F

GRID = np.arange(-21.0, -28.51, -0.5)
FLOOR = 0.02


def profile(e, ch, key):
    o = P.obs(e)
    tgl = F.glac_time(o, e)
    sel = (o["t"] >= P.TSAT[e]) & (o["t"] <= tgl)
    tt, tg, oi = o["t"][sel], o["tg"][sel], P.smooth(o["ni"], 61)[sel]
    s = F.ice_series(F.tagof(1, ch[(e, 1)]), e)
    m1 = P.model(F.tagof(1, ch[(e, 1)]), e)
    tg_lim = max(float(np.interp(tgl, o["t"], o["tg"])),
                 np.nan_to_num(P.glaciation_T(m1["t"], m1["tg"], m1["lwc"], P.TSAT[e]), nan=-99))
    S = P.initial_psd(e)["S_dust"]
    rows = []
    for T in GRID:
        if T < tg.min() or T > tg.max() or T < tg_lim:
            continue
        k = int(np.argmin(abs(tg - T)))
        ob = float(oi[k])
        mo = float(np.interp(tt[k], s["t"], s[key]))
        rows.append((float(T), ob, mo, ob / S))
    return rows


def gmean(x):
    x = np.asarray([v for v in x if np.isfinite(v) and v > 0])
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def main():
    ch = F.load_choice()
    out = {}
    for key in ("area", "vol", "dmax"):
        prof = {e: profile(e, ch, key) for e in P.EXPS}
        om = {e: gmean([ob / mo for T, ob, mo, _ in prof[e] if ob > FLOOR and mo > 0]) for e in P.EXPS}
        res = {"OM": om, "profiles": prof, "E": {}, "Eobs": {}, "Erange": {}}
        for e in P.MIX:
            a = P.ALONE[e]
            res["E"][e] = om[e] / om[a]
            pa = {T: n for T, ob, mo, n in prof[a] if ob > FLOOR}
            r = [n / pa[T] for T, ob, mo, n in prof[e] if ob > FLOOR and T in pa]
            res["Eobs"][e] = gmean(r)
            res["Erange"][e] = [min(prof[e][0][0], prof[e][-1][0]) if prof[e] else None,
                                max(prof[e][0][0], prof[e][-1][0]) if prof[e] else None, len(r)]
        out[key] = res
    json.dump(out, open(P.OUT / "matched_T.json", "w"), indent=1)
    A = out["area"]
    print("O/M at matched T (mode 1, area-sized; [vol, dmax]):")
    for e in P.EXPS:
        T = [r[0] for r in A["profiles"][e] if r[1] > FLOOR]
        print(f"  {e}: {A['OM'][e]:.2f} [{out['vol']['OM'][e]:.2f}, {out['dmax']['OM'][e]:.2f}]  T {max(T):.1f} to {min(T):.1f} C, n={len(T)}")
    print("NaCl effect not in the model, E = (O/M)mix/(O/M)alone [vol, dmax];  observed per-surface Eobs (n matched T)")
    for e in P.MIX:
        print(f"  {e} vs {P.ALONE[e]}: E {A['E'][e]:.2f} [{out['vol']['E'][e]:.2f}, {out['dmax']['E'][e]:.2f}];  Eobs {A['Eobs'][e]:.2f} (n={A['Erange'][e][2]})")


if __name__ == "__main__":
    main()
