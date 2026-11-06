import os as _os
from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[2])
HERE=str(_P(__file__).resolve().parent)
OUTDIR=_os.environ.get('ISKYLAB_RUNS',_os.path.join(REPO,'runs'))
import os, csv, numpy as np, pandas as pd, sys
sys.path.insert(0,_os.path.join(REPO,'scripts'))
from netCDF4 import Dataset
T=OUTDIR
mix=["Exp006","Exp007","Exp008","Exp009","Exp010","Exp018","Exp019"]
def win(exp):
    m={r["metric"]:r["value"] for r in csv.DictReader(open(f"{T}/baseline/single_comparison/metrics-{exp}.csv"))}
    return float(m["window_start_s"]),float(m["window_end_s"])
def permode(tag,exp,w,thr):
    f=f"{T}/{tag}/single_comparison/output-{exp}.nc"
    with Dataset(f) as nc:
        t=nc["time"][:]; n=nc["nwat"][:]; dd=nc["dwet"][:]
        p=nc["p"][:] if "p" in nc.variables else None; Tt=nc["t"][:] if "t" in nc.variables else None
    rho=p/(287.0*Tt) if p is not None else 1.2
    s=(t>=w[0])&(t<=w[1])
    nm=(n*(dd>thr*1e-6)).sum(axis=2)   # per kg? -> per cm3
    nm=nm*np.asarray(rho)[:,None]/1e6
    return nm[s].mean(axis=0)
rows=[]
for e in mix:
    w=win(e); r={"exp":e}
    for tag,lab in [("baseline","mix"),("comp_bg","bg"),("comp_mix_nobl","mix_nobl"),("comp_bg_nobl","bg_nobl")]:
        for thr in (1.0,2.5):
            v=permode(tag,e,w,thr); r[f"{lab}_m1_gt{thr}"]=round(float(v[0])); r[f"{lab}_m2_gt{thr}"]=round(float(v[1]))
    rows.append(r)
df=pd.DataFrame(rows); pd.set_option("display.width",300); print(df.to_string(index=False))
df.to_csv(T+"/competition_permode_claude.csv",index=False)
