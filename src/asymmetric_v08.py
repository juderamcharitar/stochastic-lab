"""Experimental frozen-parameter GJR-GARCH vs symmetric GARCH on a strictly later year.
This is not a replacement for expanding walk-forward refitting; intentionally labeled experimental.
Train on 2024 only, test on 2025. Gaussian QMLE of variance followed by Student-t residual fit.
"""
import argparse, zipfile, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar
from scipy.signal import lfilter
from scipy.stats import t as tdist
from src.tailrisk_v07 import t_thresholds


def hfilter(z,a,b,gamma=0,segments=None):
    z=np.asarray(z,float)
    if segments is None:segments=np.zeros(len(z),dtype=bool);segments[0]=True
    else:segments=np.asarray(segments,bool)
    starts=np.flatnonzero(segments);ends=np.r_[starts[1:],len(z)]
    w=1-a-b-gamma/2
    result=np.empty(len(z))
    for i,j in zip(starts,ends):
        part=z[i:j]
        x=np.empty(len(part));x[0]=1.0
        if len(part)>1:x[1:]=w+a*part[:-1]**2+gamma*np.minimum(part[:-1],0)**2
        result[i:j]=lfilter([1.],[1.,-b],x)
    return np.maximum(result,1e-8)

def fit(z, asymmetric=False):
    z=np.asarray(z,float)
    mu=float(z.mean());sd=max(float(z.std(ddof=1)),1e-8)
    standardized=(z-mu)/sd
    # Approximate discrete daily segmentation for a clipped contiguous training block.
    segments=np.zeros(len(z),dtype=bool);segments[0]=True
    def obj(x):
        a,b=x[:2];g=x[2] if asymmetric else 0.
        if a+b+g/2>=.995:return 1e7+10000*(a+b+g/2-.995)**2
        h=hfilter(standardized,a,b,g,segments)
        return float(.5*np.mean(np.log(h)+standardized**2/h))
    x0=[.06,.88,.05] if asymmetric else [.06,.90]
    bounds=[(.002,.3),(.01,.99)]+([(.0,.35)] if asymmetric else [])
    result=minimize(obj,x0,bounds=bounds,method='L-BFGS-B',options={'maxiter':110})
    a,b=result.x[:2];g=result.x[2] if asymmetric else 0.
    h=hfilter(standardized,a,b,g,segments)
    residual=standardized/np.sqrt(h)
    def t_obj(df):
        s=np.sqrt((df-2)/df)
        return -float(np.mean(tdist.logpdf(residual/s,df)-np.log(s)))
    df=float(minimize_scalar(t_obj,bounds=(2.2,50),method='bounded').x)
    return dict(mu=mu,sd=sd,alpha=float(a),beta=float(b),gamma=float(g),df=df,success=bool(result.success),n_train=len(z))

def run(returns,training_year=2024,test_year=2025,train_max=35000):
    r=returns.sort_index()
    training=r[r.index.year==training_year]
    test=r[r.index.year==test_year]
    if len(training)<10000 or len(test)<1000:raise ValueError('Insufficient year data')
    # Freeze all parameters before test year starts. Last training block, no future data.
    training=training.iloc[-train_max:]
    models={'Symmetric Student-t GARCH':fit(training.to_numpy(),False),'Asymmetric Student-t GJR':fit(training.to_numpy(),True)}
    rows=[]
    for name,f in models.items():
        # Separate days and discontinuities; never carry conditional variance across gaps.
        z=(test.to_numpy()-f['mu'])/f['sd']
        timestamps=test.index
        segments=np.r_[True,np.diff(timestamps.to_numpy(dtype='datetime64[ns]'))!=np.timedelta64(5,'m')]
        segments |= np.r_[True,timestamps.date[1:]!=timestamps.date[:-1]]
        h=hfilter(z,f['alpha'],f['beta'],f['gamma'],segments)
        v,e=t_thresholds(f['mu'],f['sd']*np.sqrt(h),f['df'])
        loss=-test.to_numpy();breach=loss>v
        q=.01;score=((loss-v)*breach/q+v)/e+np.log(e)-1
        rows.append(dict(model=name,training_year=training_year,test_year=test_year,n_test=len(test),breaches=int(breach.sum()),breach_rate=float(breach.mean()),joint_score=float(score.mean()),mean_VaR=float(np.mean(v)),mean_ES=float(np.mean(e)),**f))
    return pd.DataFrame(rows),models

def main():
    p=argparse.ArgumentParser();p.add_argument('--results-zip',default='STOCHASTIC_LAB_5YEAR_RESULTS.zip');p.add_argument('--output-dir',default='v08_outputs')
    a=p.parse_args()
    with zipfile.ZipFile(a.results_zip) as z:
        with z.open('btc_5m_returns.csv') as fd:df=pd.read_csv(fd,parse_dates=['open_time'])
    series=pd.Series(df.log_return_5m.values,index=pd.DatetimeIndex(pd.to_datetime(df.open_time,utc=True)))
    summary,models=run(series)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/'experimental_asymmetric_2025.csv',index=False)
    (out/'asymmetric_methodology.json').write_text(json.dumps({'experiment':'2024 training subset, frozen parameters, 2025 test','limitations':'Exploratory one-split experiment; frozen models rather than rolling refit; two-stage innovation t fit; no claim of improved trading returns. Training excerpt is 35000 latest 2024 returns.'},indent=2))
    print(summary.to_string(index=False))
if __name__=='__main__':main()
