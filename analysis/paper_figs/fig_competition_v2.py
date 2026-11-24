"""Fig. 12 (v1.8): competition, with the final wall settings (refitted with the WELAS inlet) and every model-WELAS
comparison seen through the WELAS inlet (dD2 = cfg.WELAS_INLET_DD2_UM2).
(a) % change in the total number of droplets caused by the NaCl alone (paired runs), baseline and sensitivity runs
    that reproduce the liquid water (0.5 <= R_q <= 2, scored through the inlet).
(b) Exp006: droplet spectrum, as WELAS would see it, without and with the NaCl (paired runs): where the droplets lost
    to competition are.
(c) Measured vs modelled change in droplets > 5 um between the mixture and the background-only experiment.
Writes runs/analysis/fig_competition_v2.png and competition_v2.csv."""
import os, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "scripts"))
import dataAnalysis_new as DA
sys.path.insert(0, str(Path(__file__).resolve().parent)); import paper_tags as PT
T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs"))
OUT = T / "analysis"; D = REPO / "iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"; DD2 = DA.cfg.WELAS_INLET_DD2_UM2
MIX = [("Exp006", "AS + coarse NaCl (703 cm$^{-3}$)", "Exp016", PT.BG, "coarse"),
       ("Exp007", "AS + coarse NaCl (2065 cm$^{-3}$)", "Exp016", PT.BG, "coarse"),
       ("Exp018", "FA + coarse NaCl (796 cm$^{-3}$)", "Exp015", PT.BG, "coarse"),
       ("Exp008", "AS + fine NaCl (1028 cm$^{-3}$)", "Exp016", PT.BG, "fine"),
       ("Exp009", "AS + fine NaCl (1588 cm$^{-3}$)", "Exp016", PT.BG, "fine"),
       ("Exp010", "AS + fine NaCl (3107 cm$^{-3}$)", "Exp016", PT.BG, "fine")]
VAR = [("hom", "homogeneous evaporation", "^"), ("p0", "evaporation of all sizes ($p$ = 0)", "s"), ("chenlamb", "Chen–Lamb bins", "h"),
       ("tau5", "$\\tau_{\\rm BL}$ = 5 s", "<"), ("tau20", "$\\tau_{\\rm BL}$ = 20 s", ">"),
       ("dTcold", "wall air 0.03 K colder", "v"), ("dTwarm", "wall air 0.03 K warmer", "*")]
COARSE, FINE, INK = "#c0392b", "#7f8c8d", "#222222"
def win(e):   # competition is measured early in the cloud (activation phase): the curated 2-5 min windows after onset
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{e}.csv"))}
    return float(m["window_start_s"]), float(m["window_end_s"])
def wfull(e):   # tests against WELAS use the whole-experiment window, as all other scores
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "final21/single_comparison" / f"metrics-{e}.csv"))}
    return float(m["window_start_s"]), float(m["window_end_s"])
def load(tag, e, w):
    with Dataset(T / tag / "single_comparison" / f"output-{e}.nc") as nc:
        t = nc["time"][:].data; s = (t >= w[0]) & (t <= w[1]); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s]
        nl = nc["nliq"][:].data[s] * rho[:, None, None] / 1e6; nw = nc["nwat"][:].data[s] * rho[:, None, None] / 1e6
        dw = DA.apply_welas_inlet({"dwet": nc["dwet"][:].data[s], "mwat": nc["mwat"][:].data[s]}, DD2)["dwet"] * 1e6
    return dict(n25=(nw * (dw > 2.5)).sum(axis=(1, 2)).mean(), act=nl.sum(axis=(1, 2)).mean(), n5=(nw * (dw > 5)).sum(axis=(1, 2)).mean(), nw=nw, dw=dw)
def obs5(e, w):
    o = pd.read_csv(D / f"iSKYLAB01-{e}-2-OPC-d-MergedW.csv", skipinitialspace=True); cols = [c for c in o.columns[1:] if c.replace(".", "").isdigit()]
    dm = np.array([float(c) for c in cols]); n = np.nan_to_num(o[cols].values.astype(float)) * 0.00984; t = o.iloc[:, 0].values
    s = (t >= w[0]) & (t <= w[1]); return n[s][:, dm > 5].sum(1).mean()
def rq(tag, e):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / tag / "single_comparison" / f"metrics-{e}.csv"))}
    return float(m["integrated_ql_model_kg_s_kg"]) / float(m["integrated_ql_obs_kg_s_kg"])
rows, sens = [], []
for e, lab, ref, bg, kind in MIX:
    w = win(e); M = load(PT.FINAL, e, w); B = load(bg, e, w)
    wf, wrf = wfull(e), wfull(ref); MF = load(PT.FINAL, e, wf); RF = load(PT.FINAL, ref, wrf)
    rows.append(dict(exp=e, lab=lab, kind=kind, ref=ref, act_mix=M["act"], act_bg=B["act"], change=100 * (M["act"] / B["act"] - 1),
                     Rq=rq(PT.FINAL, e), obs5=100 * (obs5(e, wf) / obs5(ref, wrf) - 1), mod5=100 * (MF["n5"] / RF["n5"] - 1),
                     obsN5=obs5(e, wf), modN5=MF["n5"], obsN5ref=obs5(ref, wrf), modN5ref=RF["n5"], pair5=100 * (M["n5"] / B["n5"] - 1), pair25=100 * (M["n25"] / B["n25"] - 1)))
for e in ["Exp006", "Exp007", "Exp018", "Exp008"]:       # sensitivity pairs at the v1.9 settings
    w = win(e)
    for v in PT.ROB_VARIANTS:
        pm, pb = T / PT.ROB.format(v=v, k="mix") / "single_comparison", T / PT.ROB.format(v=v, k="bg") / "single_comparison"
        if not ((pm / f"output-{e}.nc").exists() and (pb / f"output-{e}.nc").exists() and (pm / f"metrics-{e}.csv").exists()): continue
        am, ab = load(PT.ROB.format(v=v, k="mix"), e, w)["act"], load(PT.ROB.format(v=v, k="bg"), e, w)["act"]
        if ab < 10: continue
        q = rq(PT.ROB.format(v=v, k="mix"), e); sens.append(dict(exp=e, variant=v, change=100 * (am / ab - 1), Rq=q, ok=0.5 <= q <= 2))
df = pd.DataFrame(rows); S = pd.DataFrame(sens); df.to_csv(OUT / "competition_v2.csv", index=False); S.to_csv(OUT / "competition_v2_sens.csv", index=False)
print(df.round(1).to_string(index=False)); print(S.round(2).to_string(index=False))

fig = plt.figure(figsize=(13, 5.0)); gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.9], wspace=0.32)
# (a)
a = fig.add_subplot(gs[0]); y = np.arange(len(df))[::-1].astype(float); y[3:] -= 0.5
a.axvspan(-100, 0, color="#f6e3e0", zorder=0); a.axvline(0, color="k", lw=0.8)
for yi, r in zip(y, df.itertuples()):
    s = S[(S.exp == r.exp) & S.ok]
    for v in s.itertuples():
        mk = dict((k, m) for k, _, m in VAR)[v.variant]; a.scatter(v.change, yi, marker=mk, s=40, color="0.65", edgecolor="0.3", lw=0.4, zorder=3)
    a.scatter(r.change, yi, marker="D", s=70, color=COARSE if r.kind == "coarse" else FINE, edgecolor="k", lw=0.6, zorder=4)
    a.text(r.change, yi + 0.3, f"{r.change:+.0f} %", ha="center", fontsize=8.5, weight="bold", color=INK)
a.set_yticks(y); a.set_yticklabels(df.lab, fontsize=8.5); a.set_xlim(-70, 30); a.set_ylim(y.min() - 0.7, y.max() + 0.8)
a.set_xlabel("Change in the total number of droplets due to the NaCl (%)")
a.text(-68, y.max() + 0.55, "fewer droplets: competition", fontsize=8, color="#a33", va="center")
a.set_title("(a) Does adding NaCl reduce the number of droplets?\nModel: each experiment run without and with its NaCl", loc="left", fontsize=9.5)
a.scatter([], [], marker="D", s=50, color=COARSE, edgecolor="k", lw=0.6, label="baseline, coarse NaCl"); a.scatter([], [], marker="D", s=50, color=FINE, edgecolor="k", lw=0.6, label="baseline, fine NaCl")
used = set(S[S.ok].variant)
for k, l, m in VAR:
    if k in used: a.scatter([], [], marker=m, s=36, color="0.65", edgecolor="0.3", lw=0.4, label=l)
a.legend(frameon=False, fontsize=7.3, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, title="sensitivity runs that also reproduce the liquid water (grey):", title_fontsize=7.3)
# (b) the droplets lost from the background aerosol, split into those WELAS would and would not count, and the
#     NaCl particles WELAS would count (per mode, window means, paired runs, all sizes)
def modes(tag, e, w):
    with Dataset(T / tag / "single_comparison" / f"output-{e}.nc") as nc:
        t = nc["time"][:].data; s = (t >= w[0]) & (t <= w[1]); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s][:, None, None] / 1e6
        nw = nc["nwat"][:].data[s] * rho; nl = nc["nliq"][:].data[s] * rho
        di = DA.apply_welas_inlet({"dwet": nc["dwet"][:].data[s], "mwat": nc["mwat"][:].data[s]}, DD2)["dwet"]
    return [dict(act=nl[:, k].sum(-1).mean(), seen=(nw[:, k] * (di[:, k] > 2.5e-6)).sum(-1).mean()) for k in range(nw.shape[1])]
b = fig.add_subplot(gs[1]); BX = []
for e, lab, ref, bg, kind in MIX[:3]:
    w = win(e); A = modes(bg, e, w)[0]; Mx = modes(PT.FINAL, e, w)
    lost = A["act"] - Mx[0]["act"]; ls = A["seen"] - Mx[0]["seen"]
    BX.append(dict(exp=e, lost_seen=ls, lost_unseen=lost - ls, nacl_seen=Mx[1]["seen"], net=Mx[0]["seen"] + Mx[1]["seen"] - A["seen"], before=A["seen"]))
BX = pd.DataFrame(BX); BX.to_csv(OUT / "competition_v2_visibility.csv", index=False); print(BX.round(0).to_string(index=False))
x = np.arange(len(BX)); h = 0.36
b.bar(x - h / 2, BX.lost_unseen, h, color="#9ecae1", label="lost, not counted by WELAS")
b.bar(x - h / 2, BX.lost_seen, h, bottom=BX.lost_unseen, color="#1f6fb4", label="lost, counted by WELAS")
b.bar(x + h / 2, BX.nacl_seen, h, color="#d9822b", label="NaCl particles counted by WELAS")
for xi, r in zip(x, BX.itertuples()):
    b.text(xi - h / 2, r.lost_unseen + r.lost_seen + 15, f"{100 * r.lost_unseen / (r.lost_unseen + r.lost_seen):.0f} %\nnot\ncounted", ha="center", fontsize=7.3, color="0.3")
    b.text(xi, -95, f"WELAS would\nsee {100 * r.net / r.before:+.0f} %", ha="center", va="top", fontsize=7.8, weight="bold")
b.set_xticks(x); b.set_xticklabels(["Exp006", "Exp007", "Exp018"]); b.set_ylim(0, 1000); b.set_ylabel("Droplets or particles (cm$^{-3}$)")
b.tick_params(axis="x", pad=2); b.spines[["top", "right"]].set_visible(False)
b.legend(frameon=False, fontsize=7.8, loc="upper left", bbox_to_anchor=(0.0, -0.2))
b.set_title("(b) Why WELAS cannot see it: droplets lost to\ncompetition vs NaCl particles added (coarse NaCl)", loc="left", fontsize=9.5)
# (c)
c = fig.add_subplot(gs[2]); lim = (-20, 260); c.plot(lim, lim, color="k", lw=0.8); c.axhline(0, color="0.8", lw=0.6); c.axvline(0, color="0.8", lw=0.6)
for r in df.itertuples():
    c.scatter(r.obs5, r.mod5, marker="o", s=55, color=COARSE if r.kind == "coarse" else FINE, edgecolor="k", lw=0.5, zorder=3)
    c.annotate("Exp" + r.exp[3:], (r.obs5, r.mod5), xytext=(-6, 6) if r.exp == "Exp006" else (6, -3), textcoords="offset points", fontsize=7.5, ha="right" if r.exp == "Exp006" else "left")
c.set_xlim(*lim); c.set_ylim(*lim); c.set_aspect("equal"); c.set_xlabel("Measured by WELAS (%)"); c.set_ylabel("Model, seen through the inlet (%)")
c.spines[["top", "right"]].set_visible(False)
c.set_title("(c) Change in droplets > 5 µm, mixture\nvs the background-only experiment", loc="left", fontsize=9.5)
c.text(0.98, 0.03, "includes all differences between\nthe experiments, not only the NaCl", transform=c.transAxes, ha="right", fontsize=7.3, color="0.35")
fig.savefig(OUT / "fig_competition_v2.png", dpi=200, bbox_inches="tight")
