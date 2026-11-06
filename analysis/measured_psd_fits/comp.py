import os as _os
from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[2])
HERE=str(_P(__file__).resolve().parent)
OUTDIR=_os.environ.get('ISKYLAB_RUNS',_os.path.join(REPO,'runs'))
import numpy as np, pandas as pd, os
exec(open(_os.path.join(REPO,'analysis','wall_water_check','wall.py')).read().split("rows=[]")[0])
D=REPO+"/iSKYLAB-data/Datasets-V2"
info={}
rows=[]
for e,aer in [("005","AS*"),("011","AS"),("016","AS"),("006","AS+NaCl"),("007","AS+NaCl"),("008","AS+NaCl"),("009","AS+NaCl"),("010","AS+NaCl"),("013","NaCl"),("014","NaCl"),("015","FA"),("018","FA+NaCl"),("019","FA+NaCl")]:
    o=pd.read_csv(f"{D}/Timeseries-During-Expansion/iSKYLAB01-Exp{e}-2-OPC-d-MergedW.csv",skipinitialspace=True)
    cols=[c for c in o.columns[1:] if c.replace(".","").isdigit()]; Dm=np.array([float(c) for c in cols]); n=np.nan_to_num(o[cols].values.astype(float))*0.00984
    t=o.iloc[:,0].values
    N1=n[:,Dm>1].sum(1); N25=n[:,Dm>2.5].sum(1)
    sm=lambda x: pd.Series(x).rolling(30,center=True,min_periods=5).mean().values
    N1s,N25s=sm(N1),sm(N25)
    # PSD: count particles > 0.5 um dry in initial merged PNSD
    try:
        m=pd.read_csv(f"{D}/Initial-PNSD/iSKYLAB01-Exp{e}-3-InitialPNSD-Mrg.csv",skipinitialspace=True)
        cand=[c for c in m.columns]
        dcol=cand[0]; ncol=cand[1]
        dd=m[dcol].values.astype(float); nn=m[ncol].values.astype(float)
        dl=np.gradient(np.log10(dd)); Ntot=np.nansum(nn*dl); Nc=np.nansum((nn*dl)[dd>0.3])
    except Exception as ex: Ntot=Nc=np.nan; cand=str(ex)
    rows.append(dict(exp=e,aer=aer,N1_peak=round(np.nanmax(N1s)),N25_peak=round(np.nanmax(N25s)),PNSD_Ntot=round(Ntot) if np.isfinite(Ntot) else None,PNSD_N_gt0p3um=round(Nc,1) if np.isfinite(Nc) else None))
print(pd.DataFrame(rows).to_string(index=False)); print(cand if isinstance(cand,list) else cand)
