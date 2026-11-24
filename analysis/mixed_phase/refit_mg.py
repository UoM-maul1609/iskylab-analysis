"""Rescore the wk3c wall offsets with droplets counted per mg of air (17 Nov 2026).
Same rules as refit_score.py --prefix wk3c --constrain, except that the droplets above the
pre-cloud count are computed per mg of air (number concentration / air density), so that the
fall in air density during the expansion does not appear as a loss of droplets.
Writes paper/refit_scores_wk3c_mg_constrained.csv and refit_choice_wk3c_mg_constrained.txt."""
import csv
import os
import re
import numpy as np
os.environ.setdefault("ISKYLAB_TSAT", "hygrometer")
import paper_figs as P

ACCEPT, SENSOR = 0.3, 0.15


def candidates(e, mode):
    out = {}
    for d in P.RUNS.iterdir():
        mm = re.fullmatch(rf"wk3c_m{mode}_([pm])(\d{{3}})", d.name)
        if mm and (d / "single_comparison" / f"output-{e}.nc").exists():
            out[(1 if mm.group(1) == "p" else -1) * int(mm.group(2)) / 1000] = d.name
    return dict(sorted(out.items()))


rows, best, acc = [], {}, {}
for e in P.EXPS:
    o = P.obs(e)
    sd, si = P.windows(o, e)
    td = o["t"][sd]
    rho = o["p"] * 100 / (287.05 * (o["tg"] + 273.15))
    ndo = o["nd"] / rho
    ond = max(float(np.nanmean(ndo[sd]) - np.nanmean(ndo[o["pre"]])), 0.0)
    olwc = float(np.nanmean(o["lwc"][sd])); otw = float(np.nanmean(o["rhw"][sd]) - 100.0)
    dtw = float(np.nanmean((o["tw"] - o["tg"])[o["t"] >= P.TSAT[e]]))
    lo, hi = min(0.0, dtw) - SENSOR, max(0.0, dtw) + SENSOR
    for mode in (0, 1):
        sc = []
        for off, tag in candidates(e, mode).items():
            if not (lo - 1e-9 <= off <= hi + 1e-9):
                continue
            m = P.model(tag, e)
            ndm = m["nd"] / m["rhoa"]
            nd = float(np.mean(np.interp(td, m["t"], ndm))) - float(np.interp(0.0, m["t"], ndm))
            lwc = P.mmean(m, "lwc_welas", td); tw = P.mmean(m, "rhw_tot", td) - 100.0
            if e in P.MIX:
                s = abs(np.log(max(tw, 0.1) / max(otw, 0.1))) + abs(np.log((max(nd, 0) + 20) / (ond + 20)))
            else:
                s = abs(np.log(max(nd, 0.1) / ond)) + abs(np.log(max(lwc, 1e-5) / olwc))
            sc.append((s, off, tag))
            rows.append([e, mode, off, tag, f"{nd:.1f}", f"{ond:.1f}", f"{lwc:.4f}", f"{olwc:.4f}", f"{tw:.2f}", f"{otw:.2f}", f"{s:.3f}"])
        sc.sort(); best[(e, mode)] = sc[0]; acc[(e, mode)] = [x for x in sc if x[0] <= sc[0][0] + ACCEPT]
with open(P.OUT / "refit_scores_wk3c_mg_constrained.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["exp", "mode", "dTBL_K", "tag", "Nd_mod_per_mg", "Nd_obs_per_mg", "LWCwelas_mod", "LWC_obs", "TWexcess_mod_pct", "TWexcess_obs_pct", "score"])
    w.writerows(rows)
with open(P.OUT / "refit_choice_wk3c_mg_constrained.txt", "w") as fh:
    fh.write("# exp mode dTBL_K   (droplets per mg of air; refit_mg.py)\n")
    for (e, mode), (s, off, tag) in sorted(best.items()):
        fh.write(f"{e} {mode} {off}\n")
old = {}
for line in open(P.OUT / "refit_choice_wk3c_constrained.txt"):
    if line.strip() and not line.startswith("#"):
        a, b, c = line.split(); old[(a, int(b))] = float(c)
for (e, mode), (s, off, tag) in sorted(best.items()):
    r = [x for x in rows if x[3] == tag and x[0] == e][0]
    print(f"{e} m{mode}: new {off:+.2f} (was {old[(e, mode)]:+.2f}); N per mg mod/obs {r[4]}/{r[5]}; TW {r[8]}/{r[9]}; acceptable {[x[1] for x in acc[(e, mode)]]}")
