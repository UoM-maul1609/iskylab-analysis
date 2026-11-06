import os as _os
from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[2])
HERE=str(_P(__file__).resolve().parent)
OUTDIR=_os.environ.get('ISKYLAB_RUNS',_os.path.join(REPO,'runs'))
import os, csv, numpy as np, pandas as pd, sys
sys.path.insert(0,_os.path.join(REPO,'scripts'))
import readModel, dataAnalysis_new as d
T=OUTDIR
mix=["Exp006","Exp007","Exp008","Exp009","Exp010","Exp018","Exp019"]
refs=["Exp011","Exp015","Exp016","Exp005","Exp013","Exp014"]
def win(exp):
    m={r["metric"]:r["value"] for r in csv.DictReader(open(f"{T}/baseline/single_comparison/metrics-{exp}.csv"))}
    return float(m["window_start_s"]),float(m["window_end_s"])
def obsN(exp,thr=2.5):
    f=f"{REPO}/iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion/iSKYLAB01-{exp}-2-OPC-d-MergedW.csv"
    o=pd.read_csv(f,skipinitialspace=True); cols=[c for c in o.columns[1:] if c.replace(".","").isdigit()]
    dm=np.array([float(c) for c in cols]); n=np.nan_to_num(o[cols].values.astype(float))*0.00984
    return o.iloc[:,0].values.astype(float), n[:,dm>thr].sum(1)
def modN(tag,exp,w):
    f=f"{T}/{tag}/single_comparison/output-{exp}.nc"
    if not os.path.exists(f): return np.nan,np.nan
    m=readModel.readData(f,modelStr=exp)[exp]; t=np.asarray(m["time"])
    n25=d._model_bulk_moments_above(m,2.5)["number_cm3"]; na=np.asarray(m["ndrop"])*np.asarray(m["rhoa"])/1e6
    s=(t>=w[0])&(t<=w[1]); return np.nanmean(n25[s]),np.nanmean(na[s])
rows=[]
for e in mix+refs:
    w=win(e); to,no=obsN(e); obs=np.nanmean(no[(to>=w[0])&(to<=w[1])])
    r=dict(exp=e,window=f"{w[0]/60:.1f}-{w[1]/60:.1f}",obs_N25=round(obs))
    for tag in (["baseline","comp_bg","comp_nacl","comp_mix_nobl","comp_bg_nobl","comp_nacl_nobl"] if e in mix else ["baseline","nobl"]):
        a,b=modN(tag,e,w); r[tag+"_N25"]=round(a) if np.isfinite(a) else None; r[tag+"_Nact"]=round(b) if np.isfinite(b) else None
    rows.append(r)
df=pd.DataFrame(rows); pd.set_option("display.width",300); print(df.to_string(index=False))
df.to_csv(T+"/competition_windowmean_claude.csv",index=False)
