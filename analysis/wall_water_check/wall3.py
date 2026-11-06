import os as _os
from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[2])
HERE=str(_P(__file__).resolve().parent)
OUTDIR=_os.environ.get('ISKYLAB_RUNS',_os.path.join(REPO,'runs'))
import numpy as np, pandas as pd
exec(open(_os.path.join(REPO,'analysis','wall_water_check','wall.py')).read().split("rows=[]")[0])
# dew point from q
from scipy.optimize import brentq
rows=[]
for e in ["005","006","007","008","009","010","011","013","014","015","016","018","019"]:
    t,p,Tg,Tw,qm,lw=load(e); ok=np.isfinite(qm)
    q0=np.nanmean(qm[t<0])
    # smooth loss over 60 s
    sm=pd.Series(np.where(ok,qm,np.nan)).rolling(61,center=True,min_periods=20).mean().values
    loss=q0-sm
    # pressure effect check: qm should be const if no loss; use threshold 0.03 g/kg
    for thr in (0.03,0.05):
        i=np.argmax(loss>thr*1e-3)
        Teff=brentq(lambda T: qs(T,p[i])-sm[i],230,300)
        rows.append(dict(exp=e,thr=thr,t_loss=round(t[i]/60,1),Tg=round(Tg[i]-273.15,2),Tw=round(Tw[i]-273.15,2),Tdew_q=round(Teff-273.15,2),
            Tdew_minus_Tw=round(Teff-Tw[i],2),Tdew_minus_Tg=round(Teff-Tg[i],2)))
df=pd.DataFrame(rows); pd.set_option("display.width",250); print(df.to_string(index=False))
for thr in (0.03,0.05):
    d=df[df.thr==thr]; print(thr, "mean Td-Tw=%.2f sd=%.2f ; mean Td-Tg=%.2f"%(d.Tdew_minus_Tw.mean(),d.Tdew_minus_Tw.std(),d.Tdew_minus_Tg.mean()))
