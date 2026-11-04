"""Chronological VaR / ES validation on *contiguous* 5-minute log returns.
Loss convention: L = -log return; positive VaR and ES denote positive loss levels.
The forecasting models are deliberately simple baselines and not trading strategies.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import lfilter
from scipy.stats import norm,t as tdist
from src.multiyear import load_history

Q=.01

def garch_variances(z,a,b,initial=1.0):
    """h_t = (1-a-b) + a*z_(t-1)^2 + b*h_(t-1)."""
    z=np.asarray(z,dtype=float)
    if len(z)==0:return np.array([])
    w=1-a-b
    x=np.empty(len(z));x[0]=w+a*1.0
    x[1:]=w+a*z[:-1]**2
    return lfilter([1.],[1.,-b],x,zi=[b*(initial-1.0) + b])[0]

def fit_garch(train, segment_starts=None):
    train=np.asarray(train,float)
    mu=float(train.mean());sd=max(float(train.std(ddof=1)),1e-10)
    z=(train-mu)/sd
    if segment_starts is None:
        segment_starts=np.array([0], dtype=int)
    else:
        segment_starts=np.unique(np.asarray(segment_starts,dtype=int))
        if len(segment_starts)==0 or segment_starts[0]!=0: segment_starts=np.r_[0,segment_starts]
    endpoints=np.r_[segment_starts[1:],len(z)]
    def segmented_h(a,b):
        # Reset to unconditional variance at the start of each contiguous run.
        return np.concatenate([garch_variances(z[i:j],a,b,initial=1.0) for i,j in zip(segment_starts,endpoints)])
    # bounded deterministic QMLE, optimized in standardized space
    def objective(x):
        a,b=x
        if a<=0 or b<0 or a+b>=.997:return 1e12
        h=np.maximum(segmented_h(a,b),1e-8)
        return .5*float(np.mean(np.log(h)+z*z/h))
    res=minimize(objective,[.065,.90],method='L-BFGS-B',bounds=[(.001,.35),(.001,.995)],options={'maxiter':80,'ftol':1e-8})
    a,b=np.asarray(res.x,float)
    if a+b>=.997:a,b=.065,.90
    h=np.maximum(segmented_h(a,b),1e-8)
    resid=z/np.sqrt(h)
    if len(resid)>30000:resid=resid[np.linspace(0,len(resid)-1,30000).astype(int)]
    def obj(df):
        s=np.sqrt((df-2)/df)
        return -float(np.mean(tdist.logpdf(resid/s,df)-np.log(s)))
    df=float(minimize_scalar(obj,bounds=(2.2,50),method='bounded',options={'xatol':.08}).x)
    return {'mu':mu,'sd':sd,'alpha':float(a),'beta':float(b),'df':df,'converged':bool(res.success)}

def normal_thresholds(mu,sigma,q=Q):
    z=norm.ppf(q)
    return -mu-sigma*z, -mu+sigma*norm.pdf(z)/q

def t_thresholds(mu,sigma,df,q=Q):
    x=tdist.ppf(q,df);normfactor=np.sqrt((df-2)/df)
    left_es=-(df+x*x)/(df-1)*tdist.pdf(x,df)/q*normfactor
    return -mu-sigma*normfactor*x, -mu-sigma*left_es

def empirical_thresholds(train,q=Q):
    loss=-np.asarray(train)
    v=float(np.quantile(loss,1-q))
    tail=loss[loss>=v]
    return v,float(tail.mean()) if len(tail) else np.nan

def rolling_forecasts(r,train_days=90,refit_days=30,q=Q):
    if not isinstance(r,pd.Series) or not isinstance(r.index,pd.DatetimeIndex):raise TypeError('Expected UTC datetime-indexed 5m returns')
    if not r.index.is_monotonic_increasing or r.index.has_duplicates:raise ValueError('Returns must be sorted and unique')
    # Missing candles are allowed, but returns spanning missing intervals must have been removed by the loader.
    # Train/evaluate conditional volatility independently on each contiguous run.
    groups=[g for _,g in r.groupby(r.index.date)]
    if len(groups)<train_days+2:raise ValueError('Insufficient observed training and test days')
    rows=[];fits=[];fit=None;day_start=None
    for day_i in range(train_days,len(groups)):
        g=groups[day_i]
        if fit is None or day_i==train_days or (day_i-train_days)%refit_days==0:
            training=pd.concat(groups[max(0,day_i-train_days):day_i])
            train=training.to_numpy()
            gaps=training.index.to_series().diff().ne(pd.Timedelta(minutes=5)).to_numpy()
            fit=fit_garch(train,segment_starts=np.flatnonzero(gaps))
            gaussian_iid=normal_thresholds(float(train.mean()),float(train.std(ddof=1)),q)
            empirical=empirical_thresholds(train,q)
            # Initialize GARCH state by filtering only past observations.
            # Begin every evaluation day with unconditional variance; do not carry state over gaps.
            day_start=1.0
            fits.append({'date':str(g.index[0].date()),'training_observations':len(train),'evaluation_day_index':day_i,**fit})
        y=g.to_numpy();z=(y-fit['mu'])/fit['sd']
        a,b=fit['alpha'],fit['beta'];w=1-a-b
        gaps=g.index.to_series().diff().ne(pd.Timedelta(minutes=5)).to_numpy()
        h=np.empty(len(y),dtype=float)
        # Each five-minute forecast depends only on earlier observed returns.
        # Reset at the beginning of every calendar day and after missing intervals.
        previous_h=1.0
        for k in range(len(y)):
            if k==0 or gaps[k]:
                h[k]=1.0
            else:
                h[k]=w+a*z[k-1]**2+b*previous_h
            previous_h=h[k]
        sigma=fit['sd']*np.sqrt(np.maximum(h,1e-12))
        forecasts={
            'Gaussian IID':(np.full(len(y),gaussian_iid[0]),np.full(len(y),gaussian_iid[1])),
            'Empirical':(np.full(len(y),empirical[0]),np.full(len(y),empirical[1])),
            'GARCH Gaussian':normal_thresholds(fit['mu'],sigma,q),
            'GARCH Student-t':t_thresholds(fit['mu'],sigma,fit['df'],q)
        }
        for name,(v,e) in forecasts.items():
            for ts,loss,vi,es in zip(g.index,-y,np.atleast_1d(v),np.atleast_1d(e)):
                rows.append((ts,str(ts.date()),name,float(loss),float(vi),float(es)))
    res=pd.DataFrame(rows,columns=['timestamp','date','model','loss','VaR99','ES99'])
    res['breach']=(res.loss>res.VaR99).astype(int)
    res['es_identification']=res.breach*(res.loss-res.ES99)/q
    res['pinball']=np.where(-res.loss < -res.VaR99,q-1,q)*(-res.loss+res.VaR99)
    return res,pd.DataFrame(fits)

def block_bootstrap_es(g,draws=1500,seed=42,q=Q):
    """Day-cluster bootstrap CI for unconditional ES exceedance residual.
    Diagnostic CI, not a formal independently calibrated ES hypothesis test.
    """
    daily=g.groupby('date').agg(ident_sum=('es_identification','sum'),count=('loss','size'),breaches=('breach','sum'))
    if len(daily)<3:return [np.nan,np.nan]
    x=daily.ident_sum.to_numpy();n=daily['count'].to_numpy()
    rng=np.random.default_rng(seed);stats=[]
    for _ in range(draws):
        ids=rng.integers(0,len(daily),len(daily))
        stats.append(float(x[ids].sum()/n[ids].sum()))
    return np.quantile(stats,[.025,.975]).tolist()

def validate_es(detail,draws=1500,seed=42,q=Q):
    out=[]
    for j,(name,g) in enumerate(detail.groupby('model')):
        breaches=g[g.breach==1]
        ci=block_bootstrap_es(g,draws,seed+j,q)
        out.append({'model':name,'n':len(g),'days':int(g.date.nunique()),'breaches':len(breaches),'exceedance_rate':float(g.breach.mean()), 'mean_VaR99':float(g.VaR99.mean()),'mean_ES99':float(g.ES99.mean()),'realized_breach_loss':float(breaches.loss.mean()) if len(breaches) else np.nan,'mean_es_identification':float(g.es_identification.mean()),'es_identification_ci_low':float(ci[0]),'es_identification_ci_high':float(ci[1]),'mean_pinball':float(g.pinball.mean()),'sparse_tail_warning':len(breaches)<100 or g.date.nunique()<30})
    return pd.DataFrame(out).sort_values('mean_pinball')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='data/monthly_5m');ap.add_argument('--output-dir',default='v07_outputs')
    ap.add_argument('--train-days',type=int,default=90);ap.add_argument('--refit-days',type=int,default=30)
    ap.add_argument('--allow-unverified',action='store_true');ap.add_argument('--allow-gaps',action='store_true');ap.add_argument('--bootstraps',type=int,default=1500)
    a=ap.parse_args()
    r,coverage=load_history(a.data_dir,not a.allow_unverified,strict=not a.allow_gaps)
    detail,fits=rolling_forecasts(r,a.train_days,a.refit_days)
    summary=validate_es(detail,draws=a.bootstraps)
    out=Path(a.output_dir);out.mkdir(exist_ok=True,parents=True)
    detail.to_csv(out/'var_es_observations.csv',index=False)
    fits.to_csv(out/'refits.csv',index=False)
    summary.to_csv(out/'var_es_validation.csv',index=False)
    (out/'coverage.json').write_text(json.dumps(coverage,indent=2))
    (out/'es_methodology.json').write_text(json.dumps({'loss':'negative 5-minute log return','quantile':.99,'coverage':'one-step VaR exceedance','ES':'mean loss conditional on exceeding 99% quantile','ES_identification':'I(loss > VaR) * (loss - ES) / 0.01; expected zero under jointly calibrated VaR/ES','uncertainty':'resample complete calendar-day blocks; descriptive confidence intervals; dependence across days persists','limits':'ES identification alone is not a full conditional ES backtest; many tail exceedances and tests are required; parameter refits every N days; no transaction costs; innovation t fit is two-stage'},indent=2))
    print(summary.to_string(index=False));print('SAVED',out)
if __name__=='__main__':main()
