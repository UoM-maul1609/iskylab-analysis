"""Collect scores from batchRuns_v2.sh output into CSV tables.

Usage:  python3 collect_sensitivity.py            (reads cfg.OUTPUT_ROOT/<tag>/single_comparison)
Writes: <OUTPUT_ROOT>/sensitivity_summary.csv     one row per (tag, experiment): all metrics-*.csv values
        <OUTPUT_ROOT>/competition_summary.csv      peak/plateau droplet numbers for the competition set
        <OUTPUT_ROOT>/threshold_sensitivity.csv    N NRMSE vs counting threshold (baseline runs)
"""
import csv, glob, os
from pathlib import Path
import numpy as np
import iskylab_config as cfg
import readModel
import dataAnalysis_new as d
import experiment_metadata as meta

ROOT = cfg.OUTPUT_ROOT
TAGS = sorted(p.name for p in ROOT.iterdir() if (p / "single_comparison").is_dir())

# ---------------------------------------------------------------- 1. all metrics
rows = []
for tag in TAGS:
    for f in sorted((ROOT / tag / "single_comparison").glob("metrics-Exp*.csv")):
        exp = f.stem.replace("metrics-", "")
        with f.open() as h:
            m = {r["metric"]: r["value"] for r in csv.DictReader(h)}
        m.update(tag=tag, experiment=exp)
        rows.append(m)
if rows:
    keys = ["tag", "experiment"] + sorted({k for r in rows for k in r} - {"tag", "experiment"})
    with (ROOT / "sensitivity_summary.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {ROOT/'sensitivity_summary.csv'} ({len(rows)} rows)")

# ---------------------------------------------------------------- 2. competition
def model_numbers(nc, dmin=2.5):
    mod = readModel.readData(str(nc), modelStr="x")["x"]
    t = np.asarray(mod["time"])
    nact = np.asarray(mod["ndrop"]) * np.asarray(mod["rhoa"]) / 1e6
    n25 = d._model_bulk_moments_above(mod, dmin)["number_cm3"]
    return t, nact, n25

comp_tags = ["baseline", "comp_bg", "comp_nacl", "comp_mix_nobl", "comp_bg_nobl", "comp_nacl_nobl"]
mixtures = ["Exp006", "Exp007", "Exp008", "Exp009", "Exp010", "Exp018", "Exp019"]
crow = []
for exp in mixtures:
    r = {"experiment": exp}
    for tag in comp_tags:
        nc = ROOT / tag / "single_comparison" / f"output-{exp}.nc"
        if not nc.exists():
            continue
        t, nact, n25 = model_numbers(nc)
        r[f"{tag}_Nact_peak"] = float(np.nanmax(nact))
        r[f"{tag}_N25_peak"] = float(np.nanmax(n25))
    for a, b, c in [("baseline", "comp_bg", "comp_nacl"), ("comp_mix_nobl", "comp_bg_nobl", "comp_nacl_nobl")]:
        if f"{a}_Nact_peak" in r and f"{b}_Nact_peak" in r and f"{c}_Nact_peak" in r:
            # competition index: mixture relative to the sum of the separate populations (<1 = suppression)
            r[f"{a}_Nact_ratio_to_sum"] = r[f"{a}_Nact_peak"] / (r[f"{b}_Nact_peak"] + r[f"{c}_Nact_peak"])
            r[f"{a}_N25_ratio_to_sum"] = r[f"{a}_N25_peak"] / (r[f"{b}_N25_peak"] + r[f"{c}_N25_peak"])
            r[f"{a}_bg_suppression"] = r[f"{a}_Nact_peak"] / r[f"{b}_Nact_peak"]
    crow.append(r)
if any(len(r) > 1 for r in crow):
    keys = sorted({k for r in crow for k in r}, key=lambda k: (k != "experiment", k))
    with (ROOT / "competition_summary.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=keys); w.writeheader(); w.writerows(crow)
    print(f"wrote {ROOT/'competition_summary.csv'}")

# ---------------------------------------------------------------- 3. counting threshold (baseline)
import pandas as pd
def obs_number_above(exp, thr_um):
    f = cfg.DATA_ROOT / "Datasets-V2" / "Timeseries-During-Expansion" / f"iSKYLAB01-{exp}-2-OPC-d-MergedW.csv"
    o = pd.read_csv(f, skipinitialspace=True)
    cols = [c for c in o.columns[1:] if c.replace(".", "").isdigit()]
    dm = np.array([float(c) for c in cols])
    n = np.nan_to_num(o[cols].values.astype(float)) * 0.00984   # dN/dlogD * dlogD  -> cm-3 per bin
    return o.iloc[:, 0].values.astype(float), n[:, dm > thr_um].sum(axis=1)

trow = []
base = ROOT / "baseline" / "single_comparison"
if base.is_dir():
    exps = sorted(p.stem.replace("output-", "") for p in base.glob("output-Exp*.nc"))
    data = d.load_all_data(exps)
    for exp in exps:
        obs = d._observed_bulk_series(exp, data)
        win = d._cloud_comparison_window(exp, obs)
        mod = readModel.readData(str(base / f"output-{exp}.nc"), modelStr=exp)[exp]
        tm = np.asarray(mod["time"])
        for thr in (1.5, 2.0, 2.5, 3.0):
            n_mod = d._model_bulk_moments_above(mod, thr)["number_cm3"]
            to, n_obs = obs_number_above(exp, thr)
            sel = (to >= win[0]) & (to <= win[1])
            nm = np.interp(to[sel], tm, n_mod)
            no = n_obs[sel]
            nrmse = np.sqrt(np.nanmean((nm - no) ** 2)) / np.nanmean(no)
            trow.append({"experiment": exp, "threshold_um": thr, "obs_mean": float(np.nanmean(no)),
                         "model_mean": float(np.nanmean(nm)), "nrmse": float(nrmse)})
    with (ROOT / "threshold_sensitivity.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(trow[0])); w.writeheader(); w.writerows(trow)
    print(f"wrote {ROOT/'threshold_sensitivity.csv'}")
