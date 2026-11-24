"""Score the liquid phase of every candidate wall offset and pick dT_BL per run
and freezing mode (mixed-phase paper, 16 Nov 2026).

Fit target (the ice is NOT used to choose the offset):
  dust alone (021, 025): droplet number above pre-cloud (0.39-13 um) and mean
      WELAS liquid water over the droplet window, both after the inlet operator
      (dD2 = 25 um2).   score = |ln N_ratio| + |ln LWC_ratio|
  mixtures (022, 023, 026, 027): mean total-water excess over water saturation
      (MBW vs model vapour+liquid+ice) over the droplet window, and droplet
      number above pre-cloud (WELAS sees a droplet mode only in 026).
      score = |ln TW_ratio| + |ln((N_mod + 20)/(N_obs + 20))|
      WELAS liquid water is reported but not used: in the mixtures it depends on
      where the NaCl droplets sit relative to the ~5 um inlet cut-off.
Candidates: fin_m<k>, fin_m<k>_dTp03, fin_m<k>_dTm03 (offset = final value
+/-0.03 K) and rf_m<k>_* from batchRuns_mp_refit.sh.
"Acceptable" = score within 0.3 of the best for that run and mode; the ice
ratio R is reported at the best offsets and its range over acceptable fits.

Outputs: paper/refit_scores.csv, paper/refit_choice.txt (exp mode offset)
Usage:   python3 refit_score.py [RUNS_DIR]
         python3 refit_score.py --prefix ns [RUNS_DIR]   # runs with ns fitted to the
             dust-alone experiments (tags ns_m<k>_*); writes *_ns.csv / *_ns.txt and
             the paired ratios where <tag>dust exists
"""
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

PREFIX = None
CONSTRAIN = "--constrain" in sys.argv
if CONSTRAIN:
    sys.argv.remove("--constrain")
SENSOR = 0.15   # K, thermocouple uncertainty of Twall - Tgas
if "--prefix" in sys.argv:
    i = sys.argv.index("--prefix")
    PREFIX = sys.argv[i + 1]
    del sys.argv[i:i + 2]
import paper_figs as P  # noqa: E402  (reads RUNS from sys.argv[1])
import os  # noqa: E402

FIND = {"Exp021": 0.13, "Exp022": 0.0, "Exp023": 0.0, "Exp025": 0.0, "Exp026": 0.0, "Exp027": 0.0}
RF = {"Exp021": [0.15, 0.2, 0.25, 0.3, 0.35], "Exp025": [0.03, 0.05, 0.08, 0.1, 0.15]}
RF.update({e: [0.05, 0.1, 0.2] for e in P.MIX})
ACCEPT = 0.3


def tagof(mode, off):
    v = round(off * 1000)
    return f"rf_m{mode}_{'m' if v < 0 else 'p'}{abs(v):03d}"


def candidates(e, mode):
    if PREFIX:
        c = {}
        for d in os.listdir(P.RUNS):
            head = f"{PREFIX}_m{mode}_"
            rest = d[len(head):]
            if d.startswith(head) and len(rest) == 4 and rest[0] in "mp" and rest[1:].isdigit():
                c[round((-1 if rest[0] == "m" else 1) * int(rest[1:]) / 1000, 3)] = d
        return dict(sorted(c.items()))
    b = FIND[e]
    c = {round(b, 3): f"fin_m{mode}", round(b + 0.03, 3): f"fin_m{mode}_dTp03",
         round(b - 0.03, 3): f"fin_m{mode}_dTm03"}
    for off in RF[e]:
        c.setdefault(round(off, 3), tagof(mode, off))
    return dict(sorted(c.items()))


def main():
    rows, best, accept = [], {}, {}
    for e in P.EXPS:
        o = P.obs(e)
        sd, si = P.windows(o, e)
        td, ti = o["t"][sd], o["t"][si]
        ond = max(float(np.nanmean(o["nd"][sd]) - np.nanmean(o["nd"][o["pre"]])), 0.0)
        olwc = float(np.nanmean(o["lwc"][sd]))
        otw = float(np.nanmean(o["rhw"][sd]) - 100.0)
        oni = float(np.nanmean(o["ni"][si]))
        dtw = float(np.nanmean((o["tw"] - o["tg"])[o["t"] >= P.TSAT[e]]))
        lo, hi = min(0.0, dtw) - SENSOR, max(0.0, dtw) + SENSOR
        for mode in (0, 1):
            scored = []
            for off, tag in candidates(e, mode).items():
                m = P.model(tag, e)
                if m is None:
                    continue
                nd = P.mmean(m, "nd", td) - float(np.interp(0.0, m["t"], m["nd"]))
                lwc = P.mmean(m, "lwc_welas", td)
                tw = P.mmean(m, "rhw_tot", td) - 100.0
                ni = P.mmean(m, "ni", ti)
                if e in P.MIX:
                    s = abs(np.log(max(tw, 0.1) / max(otw, 0.1))) + abs(np.log((max(nd, 0) + 20) / (ond + 20)))
                else:
                    s = abs(np.log(max(nd, 0.1) / ond)) + abs(np.log(max(lwc, 1e-5) / olwc))
                if not CONSTRAIN or lo - 1e-9 <= off <= hi + 1e-9:
                    scored.append((s, off, tag))
                rows.append([e, mode, off, tag, f"{nd:.1f}", f"{ond:.1f}", f"{lwc:.4f}", f"{olwc:.4f}",
                             f"{tw:.2f}", f"{otw:.2f}", f"{ni:.3f}", f"{oni:.3f}", f"{s:.3f}"])
            if scored:
                scored.sort()
                best[(e, mode)] = scored[0]
                accept[(e, mode)] = [x for x in scored if x[0] <= scored[0][0] + ACCEPT]

    out = P.OUT
    sfx = (f"_{PREFIX}" if PREFIX else "") + ("_constrained" if CONSTRAIN else "")
    with open(out / f"refit_scores{sfx}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["exp", "mode", "dTBL_K", "tag", "Nd_mod", "Nd_obs", "LWCwelas_mod", "LWC_obs",
                    "TWexcess_mod_pct", "TWexcess_obs_pct", "Ni_mod", "Ni_obs", "score"])
        w.writerows(rows)
    with open(out / f"refit_choice{sfx}.txt", "w") as fh:
        fh.write("# exp mode dTBL_K   (best liquid-phase fit; see refit_score.py)\n")
        for (e, mode), (s, off, tag) in sorted(best.items()):
            fh.write(f"{e} {mode} {off}\n")

    print(f"{'exp':7s} {'mode':4s} {'tag':14s} {'dTBL':>6s} {'Nd m/o':>13s} {'LWC m/o':>15s} {'TW% m/o':>12s} {'Ni m/o':>13s} {'score':>6s}")
    for r in rows:
        mark = " *" if best.get((r[0], r[1]), (None, None, None))[2] == r[3] else ""
        print(f"{r[0]:7s} {r[1]:<4d} {r[3]:14s} {r[2]:+6.3f} {r[4]:>6s}/{r[5]:<6s} {r[6]:>7s}/{r[7]:<7s} {r[8]:>5s}/{r[9]:<6s} {r[10]:>6s}/{r[11]:<6s} {r[12]:>6s}{mark}")

    print("\nIce ratio R = Ni(mixture)/Ni(dust alone), same mode, each at its best liquid fit")
    obsR = {e: P.obs(e) for e in P.EXPS}
    for mode in (0, 1):
        for e in P.MIX:
            a = P.ALONE[e]
            if (e, mode) not in best or (a, mode) not in best:
                continue
            o = obsR[e]
            ti = o["t"][P.windows(o, e)[1]]
            mix = P.model(best[(e, mode)][2], e)
            alo = P.model(best[(a, mode)][2], a)
            oa = obsR[a]
            tia = oa["t"][P.windows(oa, a)[1]]
            R = P.mmean(mix, "ni", ti) / P.mmean(alo, "ni", tia)
            Rs = [P.mmean(P.model(x[2], e), "ni", ti) / P.mmean(P.model(y[2], a), "ni", tia)
                  for x, y in itertools.product(accept[(e, mode)], accept[(a, mode)])]
            Robs = np.nanmean(o["ni"][P.windows(o, e)[1]]) / np.nanmean(oa["ni"][P.windows(oa, a)[1]])
            print(f"  mode {mode} {e}/{a}: R = {R:.3f}  (acceptable fits {min(Rs):.3f}-{max(Rs):.3f})   observed {Robs:.3f}"
                  f"   [offsets {best[(e, mode)][1]:+.3f} / {best[(a, mode)][1]:+.3f}]")
    if PREFIX:
        print("\nPaired ratio P = Ni_all(mixture)/Ni_all(same run, NaCl removed) at the chosen offsets")
        for mode in (0, 1):
            for e in P.MIX:
                if (e, mode) not in best:
                    continue
                tag = best[(e, mode)][2]
                mix, dus = P.model(tag, e), P.model(tag + "dust", e)
                if dus is None:
                    print(f"  mode {mode} {e}: {tag}dust not run yet")
                    continue
                o = obsR[e]
                ti = o["t"][P.windows(o, e)[1]]
                a, b = P.mmean(mix, "ni_all", ti), P.mmean(dus, "ni_all", ti)
                print(f"  mode {mode} {e}: P = {a / b:6.2f}  (ice {a:.3f} vs {b:.3f}; dust airborne at end "
                      f"{mix['dust_ret'][-1]:.2f} vs {dus['dust_ret'][-1]:.2f})")
    print("\nwritten", out / f"refit_scores{sfx}.csv", "and", out / f"refit_choice{sfx}.txt")


def fallout_report():
    """Compare the fall-depth runs (tags ..._H2, ..._H4) with the same run at 2.15 m."""
    import os
    runs = [d for d in sorted(os.listdir(P.RUNS)) if d.startswith("rf_m") and ("_H2" in d or "_H4" in d)]
    if not runs:
        return
    print("\nIce fall depth (chamber_height) sensitivity: same offset and mode, H = 2.15 m vs x2 / x4")
    print(f"{'exp':7s} {'tag':18s} {'Nd m/o':>13s} {'LWC m/o':>15s} {'TW% m/o':>12s} {'Ni m/o':>13s}")
    for e in P.EXPS:
        o = P.obs(e)
        sd, si = P.windows(o, e)
        td, ti = o["t"][sd], o["t"][si]
        ond = max(float(np.nanmean(o["nd"][sd]) - np.nanmean(o["nd"][o["pre"]])), 0.0)
        olwc, otw, oni = float(np.nanmean(o["lwc"][sd])), float(np.nanmean(o["rhw"][sd]) - 100), float(np.nanmean(o["ni"][si]))
        for d in runs:
            base = d.replace("_H2", "").replace("_H4", "")
            for tag in (base, d):
                m = P.model(tag, e)
                if m is None:
                    continue
                nd = P.mmean(m, "nd", td) - float(np.interp(0.0, m["t"], m["nd"]))
                print(f"{e:7s} {tag:18s} {nd:6.1f}/{ond:<6.1f} {P.mmean(m, 'lwc_welas', td):7.4f}/{olwc:<7.4f} "
                      f"{P.mmean(m, 'rhw_tot', td) - 100:5.2f}/{otw:<6.2f} {P.mmean(m, 'ni', ti):6.3f}/{oni:<6.3f}")


if __name__ == "__main__":
    main()
    if not PREFIX:
        fallout_report()
