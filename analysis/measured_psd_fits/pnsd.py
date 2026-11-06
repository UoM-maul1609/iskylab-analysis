import os as _os
from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[2])
HERE=str(_P(__file__).resolve().parent)
OUTDIR=_os.environ.get('ISKYLAB_RUNS',_os.path.join(REPO,'runs'))
import numpy as np, os, re
D=REPO+"/iSKYLAB-data/Datasets-V2/Initial-PNSD"
def f(x):
    try: return float(x)
    except: return np.nan
for e in ["005","011","016","017","006","007","008","009","010","013","014","015","018","019"]:
    L=[l.rstrip("\n").split(",") for l in open(f"{D}/iSKYLAB01-Exp{e}-3-InitialPNSD-Mrg.csv")]
    d={}
    for r in L:
        k=re.sub(r"\s+"," ",r[0].strip()); d[k]=np.array([f(x) for x in r[1:]])
    Dv=d["Dve (um)"]; dl=d["dlogDve"]; m=min(len(Dv),len(dl),*(len(v) for v in d.values())); Dv=Dv[:m]; dl=dl[:m]; d={k:v[:m] for k,v in d.items()}
    out=[]
    for sp in ["Total","AS","NaCl","FA"]:
        k=[kk for kk in d if kk.startswith(sp+" ") and "#/cc" in kk]
        if not k: continue
        n=d[k[0]]; N=lambda lo: np.nansum((n*dl)[Dv>=lo])
        if np.nansum(n)>0: out.append(f"{sp}: N={N(0):6.0f} N>0.3={N(0.3):5.1f} N>1={N(1):4.1f}  Dmed={np.exp(np.nansum(n*dl*np.log(Dv))/np.nansum(n*dl))*1000:4.0f}nm")
    extra=[kk for kk in d if kk not in ("Dve (um)","dlogDve") and "#/cc" in kk]
    print(e," | ".join(out), "| rows:",[x.split(" ")[0] for x in extra])
