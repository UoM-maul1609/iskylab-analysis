"""Competition figure (v1.9): which simulation explains WELAS, and what it says happened.
(a) Typical model-WELAS difference (whole cloud; droplets > 2.5 um, > 5 um and liquid water, through the inlet) for each
    mixture simulated with both aerosols (v19_final), the background only (v19_bg) and the NaCl only (v19_nacl).
(b) Droplets that form on the background aerosol in the 'both' simulation, relative to the background-only simulation
    (early cloud window).
Rows ordered by NaCl particles > 0.1 um per background particle (measured initial size distributions).
Needs runs/analysis/competition_three_way.csv.  Usage: python3 fig_three_way.py -> runs/analysis/fig_three_way.png"""
import os, sys, csv
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.lines import Line2D
REPO = Path(__file__).resolve().parents[2]; T = Path(os.environ.get("ISKYLAB_RUNS", REPO / "runs")); A = T / "analysis"
P = REPO / "iSKYLAB-data/Datasets-V2/Initial-PNSD"
MIX = [("Exp006", "AS", "coarse"), ("Exp007", "AS", "coarse"), ("Exp018", "FA", "coarse"), ("Exp019", "FA", "coarse"),
       ("Exp008", "AS", "fine"), ("Exp009", "AS", "fine"), ("Exp010", "AS", "fine")]
def num(x):
    try: return float(x)
    except ValueError: return np.nan
def psd(e):   # measured initial number of background and NaCl particles, total and > 0.1 um (cm-3)
    R = list(csv.reader(open(P / f"iSKYLAB01-{e}-3-InitialPNSD-Mrg.csv"))); m = min(len(r) for r in R) - 1
    d = np.array([num(x) for x in R[0][1:m + 1]]); dl = np.array([num(x) for x in R[1][1:m + 1]]); ok = np.isfinite(d) & np.isfinite(dl)
    out = {}
    for r in R[5:]:
        n = np.nan_to_num(np.array([num(x) for x in r[1:m + 1]]) * dl)
        if n[ok].sum() > 0: out["NaCl" if r[0].strip().startswith("NaCl") else "bg"] = (n[ok].sum(), n[ok & (d > 0.1)].sum())
    return out
def win(e):
    m = {r["metric"]: r["value"] for r in csv.DictReader(open(T / "baseline/single_comparison" / f"metrics-{e}.csv"))}
    return float(m["window_start_s"]), float(m["window_end_s"])
def on_bg(tag, e):
    w = win(e)
    with Dataset(T / tag / "single_comparison" / f"output-{e}.nc") as nc:
        t = nc["time"][:].data; s = (t >= w[0]) & (t <= w[1]); rho = (nc["p"][:] / (287 * nc["t"][:])).data[s] / 1e6
        return float((nc["nliq"][:].data[s][:, 0].sum(-1) * rho).mean())
tw = pd.read_csv(A / "competition_three_way.csv"); rows = []
for e, bg, kind in MIX:
    p = psd(e); ratio = p["NaCl"][1] / p["bg"][0]
    err = {r.run: 100 * (np.exp(np.sqrt(r.misfit / 3)) - 1) for r in tw[tw.exp == e].itertuples()}
    rows.append(dict(exp=e, bg=bg, kind=kind, n_bg=p["bg"][0], n_nacl=p["NaCl"][0], n_nacl01=p["NaCl"][1], ratio=ratio,
                     err_both=err["both"], err_bg=err["no NaCl"], err_nacl=err["NaCl only"],
                     supp=100 * (on_bg("v19_final", e) / on_bg("v19_bg", e) - 1)))
df = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True); df.to_csv(A / "fig_three_way.csv", index=False)
print(df.round(2).to_string(index=False))
BG_NAME = {"AS": "ammonium sulfate", "FA": "fulvic acid"}
C_BOTH, C_BG, C_NA = "#222222", "#1f6fb4", "#d9822b"
fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True, gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.08))
y = np.arange(len(df))
for i, r in df.iterrows():
    a.plot([min(r.err_both, r.err_bg, r.err_nacl), max(r.err_both, r.err_bg, r.err_nacl)], [i, i], color="0.85", lw=1, zorder=1)
    a.scatter(r.err_bg, i, s=95, marker="o", color=C_BG, zorder=3); a.scatter(r.err_nacl, i, s=95, marker="s", color=C_NA, zorder=3)
    a.scatter(r.err_both, i, s=150, marker="*", color=C_BOTH, zorder=4)
a.set_xscale("log"); a.set_xlim(3, 4000); a.set_xticks([3, 10, 30, 100, 300, 1000, 3000]); a.set_xticklabels(["3", "10", "30", "100", "300", "1000", "3000"])
a.axvspan(3, 20, color="#e8f4ea", zorder=0); a.text(3.3, len(df) - 0.45, "good agreement", fontsize=9, color="#2a7a2a", va="center")
a.set_xlabel("Typical difference between model and WELAS (%)", fontsize=11)
labs = [f"{r.exp}: {BG_NAME[r.bg]} {r.n_bg:.0f} cm$^{{-3}}$\n+ {r.kind} NaCl {r.n_nacl:.0f} cm$^{{-3}}$ ({r.ratio:.2f})" for r in df.itertuples()]
a.set_yticks(y); a.set_yticklabels(labs, fontsize=9); a.set_ylim(-0.6, len(df) - 0.2)
a.set_title("(a) Which simulation explains what WELAS measured?", loc="left", fontsize=11.5, weight="bold")
a.legend(handles=[Line2D([], [], lw=0, marker="*", ms=14, color=C_BOTH, label="both aerosols"),
                  Line2D([], [], lw=0, marker="o", ms=9, color=C_BG, label="background aerosol only"),
                  Line2D([], [], lw=0, marker="s", ms=9, color=C_NA, label="NaCl only")], frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
a.spines[["top", "right"]].set_visible(False)
for ax in (a, b): ax.axhline(2.5, color="0.6", lw=0.8, ls=":")
a.text(3800, 2.62, "coarse NaCl", ha="right", va="bottom", fontsize=9, color="0.4"); a.text(3800, 2.38, "fine NaCl", ha="right", va="top", fontsize=9, color="0.4")
SUP, ENH = "#2166ac", "#c0392b"
cols = [SUP if v <= -2 else (ENH if v >= 2 else "0.55") for v in df.supp]
b.axvspan(-140, 0, color="#e6eef7", zorder=0); b.axvspan(0, 45, color="#f8e6e3", zorder=0)
b.text(-135, len(df) - 0.45, "suppression", fontsize=10.5, weight="bold", color=SUP, va="center")
b.text(23, 4.5, "enhance-\nment\n(none\nseen)", fontsize=10, weight="bold", color=ENH, va="center", ha="center")
b.barh(y, df.supp, color=cols, height=0.55, zorder=2); b.axvline(0, color="k", lw=0.8)
for i, v in enumerate(df.supp):
    b.text(v - 3 if v <= -2 else max(v, 0) + 3, i, (f"{v:+.0f} %").replace("-", "\u2212") if abs(v) >= 0.5 else "0 %", va="center",
           ha="right" if v <= -2 else "left", fontsize=10, weight="bold", color=cols[i], zorder=3)
b.set_xlim(-140, 45); b.set_xticks([-100, -75, -50, -25, 0, 25]); b.set_xlabel("Change in droplets formed on the background aerosol\nwhen the NaCl is present (%)", fontsize=11)
b.set_title("(b) What that simulation says the NaCl did", loc="left", fontsize=11.5, weight="bold")
b.spines[["top", "right", "left"]].set_visible(False); b.tick_params(axis="y", length=0)
fig.text(0.01, -0.1, "Rows ordered by the number of NaCl particles larger than 0.1 µm per background particle (in brackets).", fontsize=9, color="0.35")
fig.suptitle("Only simulations with both aerosols explain every mixture: coarse NaCl took the cloud from the background aerosol, fine NaCl did not",
             fontsize=12.5, weight="bold", y=1.0)
fig.savefig(A / "fig_three_way.png", dpi=200, bbox_inches="tight")
