"""Fit the wall vapour-transfer velocity km to the hygrometer before cloud onset
(mixed-phase paper, step 1 of the redo, 16 Nov 2026).

Runs: wk_m1_k<NN> from batchRuns_mp_refit.sh wallfit (measured start humidity,
aerosol at t=0, BMM wall mode 2 with km = NN x 1e-4 m/s).

Target: total water expressed as relative humidity over water (vapour + condensate;
MBW frost/dew point vs model rh + (ql + qi)/qs), from 60 s after the start to the
observed cloud onset.  Score = RMS difference in percentage points.
Also reported: the model's cloud onset (first time RH over water >= 100 %) against
the observed onset, and the total water at onset.

Usage: python3 wall_score.py
"""
import os

import numpy as np

import paper_figs as P

KS = ["01", "02", "03", "05", "08"]


def onset(t, rh):
    k = np.where(rh >= 1.0)[0]
    return float(t[k[0]]) if len(k) else np.nan


def main():
    best = {}
    print(f"{'exp':7s} {'km (m/s)':>9s} {'RMS (pp)':>9s} {'TW at onset m/o (%)':>20s} {'onset m/o (min)':>16s} {'RH0 m/o (%)':>13s}")
    for e in P.EXPS:
        o = P.obs(e)
        ton = P.TSAT[e]
        sel = (o["t"] >= 60) & (o["t"] <= ton)
        rows = []
        for k in KS:
            tag = f"wk_m1_k{k}"
            if not (P.RUNS / tag / "single_comparison" / f"output-{e}.nc").exists():
                continue
            m = P.model(tag, e)
            mod = np.interp(o["t"][sel], m["t"], m["rhw_tot"])
            rms = float(np.sqrt(np.nanmean((mod - o["rhw"][sel]) ** 2)))
            tw_on_m = float(np.interp(ton, m["t"], m["rhw_tot"]))
            tw_on_o = float(np.nanmean(o["rhw"][(o["t"] > ton - 30) & (o["t"] < ton + 30)]))
            on_m = onset(m["t"], m["rh"])
            rows.append((rms, k))
            print(f"{e:7s} {int(k) * 1e-4:9.1e} {rms:9.2f} {tw_on_m:9.1f}/{tw_on_o:<9.1f} {on_m / 60:7.1f}/{ton / 60:<7.1f} "
                  f"{100 * m['rh'][0]:5.1f}/{np.nanmean(o['rhw'][o['pre']]):<5.1f}")
        if rows:
            best[e] = min(rows)
            print(f"{'':7s} best km = {int(best[e][1]) * 1e-4:.1e} m/s\n")
    if best:
        ks = [int(v[1]) * 1e-4 for v in best.values()]
        print("best km per run:", {e: f"{int(v[1]) * 1e-4:.0e}" for e, v in best.items()},
              f"| median {np.median(ks):.1e} m/s")
        with open(P.OUT / "wall_km_choice.txt", "w") as fh:
            fh.write("# exp km_m_per_s   (best fit to MBW total water before cloud onset; wall_score.py)\n")
            for e, v in best.items():
                fh.write(f"{e} {int(v[1]) * 1e-4:.1e}\n")


if __name__ == "__main__":
    main()
