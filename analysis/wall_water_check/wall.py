import os as _os
from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[2])
HERE=str(_P(__file__).resolve().parent)
OUTDIR=_os.environ.get('ISKYLAB_RUNS',_os.path.join(REPO,'runs'))
import pandas as pd, numpy as np, sys
from scipy.optimize import brentq
D=REPO+"/iSKYLAB-data/Datasets-V2/Timeseries-During-Expansion"
es=lambda T: 611.2*np.exp(17.67*(T-273.15)/(T-29.65))   # Pa, liquid
eps=0.622; cp=1005.; L=2.5e6; Rd=287.
qs=lambda T,p: eps*es(T)/(p-es(T))
paper={"005":(-0.35,80),"006":(-0.09,10),"007":(-0.1,10),"008":(-0.095,10),"009":(-0.095,10),"010":(-0.095,10),"011":(-0.1,10),"013":(-0.085,10),"014":(-0.106,10),"015":(-0.075,10),"016":(-0.08,10),"017":(-0.55,10),"018":(-0.09,10),"019":(-0.1,10)}
def load(e):
    d=pd.read_csv(f"{D}/iSKYLAB01-Exp{e}-1-MeteoCPC.csv",skipinitialspace=True); d.columns=[c.strip() for c in d.columns]
    o=pd.read_csv(f"{D}/iSKYLAB01-Exp{e}-2-OPC-d-MergedW.csv",skipinitialspace=True)
    cols=[c for c in o.columns[1:] if c.replace(".","").isdigit()]; Dm=np.array([float(c) for c in cols])*1e-6; to=o.iloc[:,0].values; n=o[cols].values.astype(float)
    n=np.nan_to_num(n)*1e6*0.00984  # per m3 per bin
    t=d["Time (Sec)"].values; p=d["Pressure (hPa)"].values*100; Tg=d["Tgw_mean (degC)"].values+273.15; Tw=d["Tww_mean (degC)"].values+273.15; Td=d["TDew (degC)"].values+273.15
    fill=lambda x: np.interp(t,t[np.isfinite(x)],x[np.isfinite(x)])
    p=fill(p); Tg=fill(Tg); Tw=fill(Tw)
    rho=p/(Rd*Tg)
    mask=Dm>2.5e-6
    LWCo=(n[:,mask]*np.pi/6*1000*Dm[mask]**3).sum(1)
    lw=np.interp(t,to,LWCo)/rho
    qm=np.where(np.isfinite(Td),qs(Td,p),np.nan)
    return t,p,Tg,Tw,qm,lw
def bl_loss(qv,Tbl,p):
    if qv<=qs(Tbl,p): return 0.
    f=lambda T: qv-qs(T,p)-cp/L*(T-Tbl)
    Ts=brentq(f,Tbl-5,Tbl+20); return qv-qs(Ts,p)
def run(t,p,Tg,Tw,q0,off,tau,alpha=1.0):
    q=q0; out=[q]; 
    for k in range(1,len(t)):
        dt=t[k]-t[k-1]; qv=min(q,qs(Tg[k],p[k])); Tbl=Tg[k]+alpha*(Tw[k]-Tg[k])+off
        fm=1-np.exp(-dt/tau); q-=fm*bl_loss(qv,Tbl,p[k]); out.append(q)
    return np.array(out)
rows=[]
for e,(off_p,tau) in paper.items():
    t,p,Tg,Tw,qm,lw=load(e)
    ok=np.isfinite(qm)
    on=t[np.argmax(lw>1e-5)] if (lw>1e-5).any() else np.nan
    q0=np.nanmean(qm[(t<0)])
    # comparison window: from onset to last valid MBW point but max onset+600s (017: onset+120)
    span=120 if e=="017" else 600
    w=ok&(t>=on)&(t<=on+span)
    if w.sum()<5: rows.append((e,on,np.nan)); continue
    tend=t[w][-1]; iend=np.argmin(abs(t-tend))
    obs_loss=q0-np.nanmean(qm[(t>tend-30)&(t<=tend)&ok])
    # mbw - qs(Tg) vs welas ql at tend
    qsg=qs(Tg[iend],p[iend]); exc=np.nanmean(qm[(t>tend-30)&(t<=tend)&ok])-qsg; lwe=np.mean(lw[(t>tend-30)&(t<=tend)])
    mod=run(t[:iend+1],p[:iend+1],Tg[:iend+1],Tw[:iend+1],q0,off_p,tau)
    mod_loss=q0-mod[-1]
    g=lambda o: (q0-run(t[:iend+1],p[:iend+1],Tg[:iend+1],Tw[:iend+1],q0,o,tau)[-1])-obs_loss
    try: off_fit=brentq(g,-3,1.5)
    except Exception: off_fit=np.nan
    rows.append(dict(exp=e,onset_min=round(on/60,1),tend_min=round(tend/60,1),q0=round(q0*1e3,3),obs_loss=round(obs_loss*1e3,3),mod_loss_paperdT=round(mod_loss*1e3,3),
        dT_paper=off_p,dT_fit=round(off_fit,3),TgTw=round(np.mean((Tg-Tw)[(t>on)&(t<tend)]),2),mbw_minus_qs=round(exc*1e3,3),welas_ql=round(lwe*1e3,3)))
pd.set_option("display.width",250); print(pd.DataFrame([r for r in rows if isinstance(r,dict)]).to_string(index=False))
