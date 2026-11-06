import os as _os
from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[2])
HERE=str(_P(__file__).resolve().parent)
OUTDIR=_os.environ.get('ISKYLAB_RUNS',_os.path.join(REPO,'runs'))
import numpy as np, pandas as pd
exec(open(_os.path.join(REPO,'analysis','wall_water_check','wall.py')).read().split("rows=[]")[0])
rows=[]
for e,(off,tau) in paper.items():
    if e in ("005","017"): continue
    t,p,Tg,Tw,qm,lw=load(e); ok=np.isfinite(qm); above=lw>1e-5
    idx=next(i for i in range(len(t)) if above[i:i+60].all()); on=t[idx]; w=ok&(t>on+60)
    obs=-np.polyfit(t[w],qm[w],1)[0]*60*1e3
    r_gas=np.mean([bl_loss(qs(Tg[k],p[k]),Tg[k]+off,p[k])/tau for k in np.where(w)[0]])*60*1e3
    rows.append(dict(exp=e,obs=round(obs,3),emul_alpha0=round(r_gas,3),ratio=round(r_gas/obs,2)))
df=pd.DataFrame(rows); print(df.to_string(index=False)); print("median ratio",df.ratio.median())
