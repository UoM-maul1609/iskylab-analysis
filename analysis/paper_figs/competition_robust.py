"""Is the modelled competition robust? Change in total activated droplets (mixture vs same run with the
background aerosol only) across model variants, with the liquid-water fit (R_q) of each mixture run.
Run after:  ./batchRuns_v2.sh robust   (baseline and comp sets must also exist)."""
import os, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ISKYLAB_RUNS", Path(__file__).resolve().parents[2] / "runs"))
OUT = T / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
EXPS = [("Exp006", "AS + coarse NaCl 703"), ("Exp007", "AS + coarse NaCl 2065"), ("Exp018", "FA + coarse NaCl 796"), ("Exp008", "AS + fine NaCl 1028")]
VARS = [("baseline", "baseline", "comp_bg", "Baseline"), ("hom", "rob_hom_mix", "rob_hom_bg", "Homogeneous evaporation"),
        ("p0", "rob_p0_mix", "rob_p0_bg", "Evaporation of all sizes ($p$=0)"), ("chenlamb", "rob_chenlamb_mix", "rob_chenlamb_bg", "Chen–Lamb bins"),
        ("tau5", "rob_tau5_mix", "rob_tau5_bg", r"$\tau_{BL}$ = 5 s"), ("tau20", "rob_tau20_mix", "rob_tau20_bg", r"$\tau_{BL}$ = 20 s"),
        ("dTcold", "rob_dTcold_mix", "rob_dTcold_bg", r"$\Delta T_{BL}$ − 0.03 K"), ("dTwarm", "rob_dTwarm_mix", "rob_dTwarm_bg", r"$\Delta T_{BL}$ + 0.03 K")]
def win(e):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{e}.csv"))}; return float(m["window_start_s"]), float(m["window_end_s"])
def act(tag, e):
    p = T / tag / "single_comparison" / f"output-{e}.nc"
    if not p.exists(): return np.nan
    w = win(e)
    with Dataset(p) as nc:
        t = nc["time"][:].data; s = (t >= w[0]) & (t <= w[1]); return float((nc["ndrop"][:].data * nc["p"][:].data / (287 * nc["t"][:].data))[s].mean() / 1e6)
def rq(tag, e):
    p = T / tag / "single_comparison" / f"metrics-{e}.csv"
    if not p.exists(): return np.nan
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(p))}
    return float(m["integrated_ql_model_kg_s_kg"]) / float(m["integrated_ql_obs_kg_s_kg"])
rows = []
for e, lab in EXPS:
    for v, tm, tb, vl in VARS:
        a, b = act(tm, e), act(tb, e)
        rows.append(dict(exp=e, label=lab, variant=v, variant_label=vl, act_mix=a, act_bg=b, change_pct=(100 * (a / b - 1) if b > 10 else np.nan), Rq_mix=rq(tm, e)))
df = pd.DataFrame(rows); df.to_csv(OUT / "competition_robust.csv", index=False)
print(df.round(2).to_string(index=False))
# figure: change in activated droplets per variant, coarse vs fine; hollow markers where liquid water is off by > factor 2
fig, ax = plt.subplots(figsize=(7.6, 4.2))
cols = {"Exp006": "#1f6fb4", "Exp007": "#4d9de0", "Exp018": "#7b52ab", "Exp008": "#d9822b"}; mk = {"Exp006": "o", "Exp007": "s", "Exp018": "D", "Exp008": "^"}
y = np.arange(len(VARS))[::-1]
for e, lab in EXPS:
    d = df[df.exp == e]
    for i, (v, *_rest) in enumerate(VARS):
        r = d[d.variant == v].iloc[0]
        if np.isnan(r.change_pct):
            ax.text(2, y[i] + {"Exp006": .18, "Exp007": .06, "Exp018": -.06, "Exp008": -.18}[e], "no cloud" if e == "Exp006" else "", fontsize=7, va="center", color="0.4"); continue
        if np.isinf(r.change_pct): ax.annotate("background alone: no droplets", (5, y[i] + {"Exp006": .18, "Exp007": .06, "Exp018": -.06, "Exp008": -.18}[e]), fontsize=6.5, color=cols[e], va="center"); continue
        good = 0.5 <= r.Rq_mix <= 2
        ax.scatter(r.change_pct, y[i] + {"Exp006": .18, "Exp007": .06, "Exp018": -.06, "Exp008": -.18}[e], marker=mk[e], s=40,
                   facecolor=cols[e] if good else "white", edgecolor=cols[e], zorder=3, label=lab if i == 0 else None)
ax.set_xlim(-100, 140); ax.axvline(0, color="k", lw=1); ax.axvspan(-100, 0, color="0.93", zorder=0)
ax.text(-2, len(VARS) - 0.45, "fewer droplets\n(competition)", ha="right", va="top", fontsize=8); ax.text(2, len(VARS) - 0.45, "more droplets", ha="left", va="top", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([v[3] for v in VARS]); ax.set_ylim(-0.6, len(VARS) - 0.2)
ax.set_xlabel("Change in total activated droplets when NaCl is added (%)"); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT / "fig_comp_robust.png", dpi=200); plt.close(fig)
co = df[(df.exp != "Exp008") & np.isfinite(df.change_pct)]
fi = df[(df.exp == "Exp008") & np.isfinite(df.change_pct) & (df.Rq_mix >= 0.5) & (df.Rq_mix <= 2)]
print("fine NaCl, runs fitting liquid water:", fi[["variant", "change_pct"]].round(1).values.tolist())
print("coarse, fitting liquid water:", co[(co.Rq_mix >= 0.5) & (co.Rq_mix <= 2)][["exp", "variant", "change_pct"]].round(1).values.tolist()); ok = co[(co.Rq_mix >= 0.5) & (co.Rq_mix <= 2)]
print(f"\ncoarse NaCl: {int((co.change_pct < 0).sum())} of {len(co)} paired cases show fewer droplets; "
      f"among those fitting liquid water within x2: {int((ok.change_pct < 0).sum())} of {len(ok)}; range {co.change_pct.min():.0f} to {co.change_pct.max():.0f} %")
