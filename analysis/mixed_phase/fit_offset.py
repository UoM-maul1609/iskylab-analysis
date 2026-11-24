"""Per-run wall offset for the cold runs (set mpoff): which dT_BL fits the droplets?

For every run and offset: droplet number through the WELAS operator and liquid
water, as window means (t_sat .. t_sat + 600 s) against WELAS, plus the NRMSE of
the droplet number over the same window, and the maximum supersaturation.
Best offset per run = smallest |log(Nd_model / Nd_obs)|, reported with the LWC
ratio so that a droplet fit with wrong liquid water is visible.

Usage: python3 fit_offset.py [RUNS_DIR]
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_mp as cmp  # noqa: E402

if len(sys.argv) > 1:
    cmp.RUNS = Path(sys.argv[1])
OFFS = [("mp_off_m030", -0.03), ("mp_off_m010", -0.01), ("mp_off_p000", 0.0), ("mp_off_p050", 0.05),
        ("mp_off_p100", 0.1), ("mp_off_p200", 0.2), ("mp_off_p300", 0.3), ("mp_off_p500", 0.5)]


def main():
    best = {}
    for exp in cmp.TSAT:
        o = cmp.obs(exp)
        td, _ = cmp.windows(o, exp)
        sel = np.isin(o["t"], td)
        ond, olwc = np.nanmean(o["nd"][sel]), np.nanmean(o["lwc"][sel])
        print(f"== {exp}: observed Nd {ond:.0f} cm-3, LWC {olwc:.3f} g m-3")
        print(f"   {'dT_BL':>6s} {'Smax%':>6s} {'Nd':>7s} {'Nd/obs':>7s} {'NRMSE':>6s} {'LWC/obs':>8s}")
        rows = []
        for tag, off in OFFS:
            m = cmp.model(tag, exp)
            if m is None:
                continue
            nd_t = np.interp(td, m["t"], m["nd"])
            lwc = cmp.wmean(m, "lwc", 0, 0, td)
            nd = float(np.mean(nd_t))
            nrmse = float(np.sqrt(np.nanmean((nd_t - o["nd"][sel]) ** 2)) / ond)
            rows.append((off, nd / ond, nrmse, lwc / olwc))
            print(f"   {off:+6.2f} {100*m['smax']:6.2f} {nd:7.1f} {nd/ond:7.2f} {nrmse:6.2f} {lwc/olwc:8.1f}")
        ok = [r for r in rows if r[1] > 0]
        if ok:
            b = min(ok, key=lambda r: abs(np.log(r[1])))
            best[exp] = b
    print("\nBest offset per run (droplet number):")
    for exp, (off, r, nrmse, lr) in best.items():
        print(f"   {exp}: dT_BL = {off:+.2f} K   Nd/obs {r:.2f}   NRMSE {nrmse:.2f}   LWC/obs {lr:.1f}")


if __name__ == "__main__":
    main()
