"""V0.9: calendar-safe, chronological GARCH vs GJR Student-t comparison.
No candle fabrication; forecasts for observation t use only observations < t.
Year/window summaries and paired day-block bootstrap scores.

Run: py -m src.robustness_v09 --returns-csv btc_5m_returns.csv --train-days 60,120,240 --refit-days 90
Preview: py -m src.robustness_v09 --returns-csv btc_5m_returns.csv --train-days 60,120 --refit-days 120 --start-year 2025
"""
import argparse,json,time,zipfile,io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import t as student_t
from scipy.signal import lfilter

Q=.01

def get_breaks(index):
    idx=pd.DatetimeIndex(index)
    if len(idx)==0:return np.empty(0,bool)
    gaps=np.r_[True,np.diff(idx.asi8)!=pd.Timedelta('5min').value]
    gaps[1:] |= np.asarray(idx.date[1:]!=idx.date[:-1])
    return gaps

def hfilter(z,a,b,g,breaks):
    """h_t=omega + (alpha*z_(t-1)^2+gamma*z_(t-1)^2*I(z<0)) + beta*h_(t-1).
    Initial h=1 for each observed calendar day or discontinuity; no future z enters forecast.
    """
    z=np.asarray(z,float);breaks=np.asarray(breaks,bool)
    if not len(z):return np.empty(0)
    if len(z)!=len(breaks) or not breaks[0]:raise ValueError('Invalid segment starts')
    if a<0 or b<0 or g<0 or a+b+g/2>=1:raise ValueError('Nonstationary parameter region')
    w=1-a-b-g/2
    out=np.empty(len(z));starts=np.flatnonzero(breaks);ends=np.r_[starts[1:],len(z)]
    for i,j in zip(starts,ends):
        count=j-i
        if count==1:out[i]=1.0;continue
        zz=z[i:j-1]
        innovation=a*zz**2+g*np.where(zz<0,zz**2,0)
        inp=w+innovation
        # scipy.signal.lfilter initial condition makes first filtered observation w+innovation[0]+b*1
        out[i]=1.0
        out[i+1:j]=lfilter([1.0],[1.0,-b],inp,zi=[b])[0]
    return np.maximum(out,1e-10)

def fit_model(training,asymmetric,max_fit_obs=100000,maxiter=55):
    if not isinstance(training,pd.Series):raise TypeError('expected Series')
    # A bounded, strictly chronological training excerpt; window labels reflect max days,
    # and actual sampled observation count is reported.
    tr=training.iloc[-max_fit_obs:]
    zraw=tr.to_numpy(dtype=float)
    if len(zraw)<1000:raise ValueError('Not enough training returns')
    mu=float(zraw.mean());scale=float(zraw.std(ddof=1))
    if scale<1e-12:raise ValueError('Near constant historical returns')
    z=(zraw-mu)/scale;breaks=get_breaks(tr.index)
    def obj(x):
        a,b=x[:2];g=x[2] if asymmetric else 0.
        if a+b+g/2 >=.995:return 1e6 + 1e5*(a+b+g/2-.995)**2
        try:h=hfilter(z,a,b,g,breaks)
        except ValueError:return 1e8
        return float(np.mean(.5*(np.log(h)+z*z/h)))
    x0=[.065,.87,.05] if asymmetric else [.065,.90]
    bounds=[(.002,.30),(.02,.98)]+([(.0,.32)] if asymmetric else [])
    opt=minimize(obj,x0,method='L-BFGS-B',bounds=bounds,options={'maxiter':maxiter,'ftol':1e-7})
    a,b=map(float,opt.x[:2]);g=float(opt.x[2]) if asymmetric else 0.0
    if a+b+g/2>=.995:raise RuntimeError('Optimizer returned nonstationary parameters')
    h=hfilter(z,a,b,g,breaks)
    resid=z/np.sqrt(h)
    def t_loss(df):
        sf=np.sqrt((df-2)/df)
        return -float(np.mean(student_t.logpdf(resid/sf,df)-np.log(sf)))
    dof=float(minimize_scalar(t_loss,bounds=(2.2,50),method='bounded',options={'xatol':.1}).x)
    return dict(mu=mu,scale=scale,alpha=a,beta=b,gamma=g,df=dof,converged=bool(opt.success),n_fit=len(tr),objective=float(opt.fun))

def thresholds(mu,sig,df,q=Q):
    x=student_t.ppf(q,df);s=np.sqrt((df-2)/df)
    var=-mu-sig*s*x
    es=-mu+sig*s*((df+x*x)/(df-1))*student_t.pdf(x,df)/q
    return var,es

def joint_score(loss,var,es,q=Q):
    # FZ-style joint score restricted to strictly positive ES.
    if np.any(np.asarray(es)<=0):raise ValueError('ES nonpositive; score is invalid')
    return ((loss-var)*(loss>var)/q+var)/es+np.log(es)-1.

def run(r,train_days=(60,120,240),refit_days=90,start_year=2021,fit_obs=100000,progress=True):
    if r.index.has_duplicates or not r.index.is_monotonic_increasing:raise ValueError('Index must be sorted and unique')
    if len(r)<10000:raise ValueError('Not enough history')
    grouped=[(str(ts),g) for ts,g in r.groupby(r.index.date)]
    first=max(train_days)
    end_rows=[];fits=[]
    for window in train_days:
        next_fit=None;params={}
        if progress:print(f'WINDOW {window} observed training days',flush=True)
        for k in range(window,len(grouped)):
            date,g=grouped[k]
            if int(date[:4])<start_year:continue
            if next_fit is None or k>=next_fit:
                train=pd.concat([v for _,v in grouped[k-window:k]])
                params={}
                for name,asymmetric in [('Symmetric Student-t GARCH',False),('Asymmetric Student-t GJR',True)]:
                    pars=fit_model(train,asymmetric,max_fit_obs=fit_obs)
                    params[name]=pars
                    fits.append(dict(date=date,model=name,train_days=window,refit_days=refit_days,train_first=str(train.index[0]),train_last=str(train.index[-1]),**pars))
                next_fit=k+refit_days
                if progress:print(f'  refitted on {date} (previous {window} days)',flush=True)
            loss=-g.to_numpy(float)
            breaks=get_breaks(g.index)
            for name,p in params.items():
                z=(g.to_numpy(float)-p['mu'])/p['scale']
                h=hfilter(z,p['alpha'],p['beta'],p['gamma'],breaks)
                v,es=thresholds(p['mu'],p['scale']*np.sqrt(h),p['df'])
                valid=np.isfinite(v)&np.isfinite(es)&(es>0)
                if not valid.all():raise ValueError('Invalid or non-positive ES forecast')
                score=joint_score(loss,v,es)
                breach=loss>v
                end_rows.append(dict(date=date,year=int(date[:4]),model=name,train_days=window,refit_days=refit_days,n=len(loss),breaches=int(breach.sum()),breach_rate=float(breach.mean()),joint_score_sum=float(score.sum()),pinball_sum=float(np.maximum(loss-v,0).sum()*Q+np.maximum(v-loss,0).sum()*(1-Q)),es_id_sum=float((breach*(loss-es)/Q).sum()),loss_beyond_var=float(np.maximum(loss-v,0).sum())))
    return pd.DataFrame(end_rows),pd.DataFrame(fits)

def summarize(daily):
    if daily.empty:return pd.DataFrame()
    rows=[]
    for (window,model,year),g in daily.groupby(['train_days','model','year']):
        n=g.n.sum();rows.append(dict(train_days=window,model=model,year=year,n=int(n),days=len(g),breaches=int(g.breaches.sum()),breach_rate=float(g.breaches.sum()/n),mean_joint_score=float(g.joint_score_sum.sum()/n),mean_es_identification=float(g.es_id_sum.sum()/n)))
    return pd.DataFrame(rows)

def paired_bootstrap(daily,draws=1500,seed=2026,block_days=7):
    """Paired moving-block bootstrap of consecutive observed dates.
    Captures some day-to-day dependence, but remains exploratory and sensitive to block length.
    """
    rng=np.random.default_rng(seed);rows=[]
    for window,g in daily.groupby('train_days'):
        p=g.pivot(index='date',columns='model',values=['joint_score_sum','n']).dropna()
        a='Asymmetric Student-t GJR';s='Symmetric Student-t GARCH'
        if a not in p.joint_score_sum or s not in p.joint_score_sum:continue
        diff=(p.joint_score_sum[a]-p.joint_score_sum[s]).to_numpy()
        counts=p.n[s].to_numpy()
        if len(diff)<4:continue
        block=min(block_days,len(diff))
        n_blocks=int(np.ceil(len(diff)/block))
        starts=rng.integers(0,len(diff)-block+1,size=(draws,n_blocks))
        sample=(starts[:,:,None]+np.arange(block)[None,None,:]).reshape(draws,-1)[:,:len(diff)]
        estimates=diff[sample].sum(axis=1)/counts[sample].sum(axis=1)
        observed=float(diff.sum()/counts.sum())
        lo,hi=np.quantile(estimates,[.025,.975])
        rows.append(dict(train_days=int(window),bootstrap_block_days=block,n_days=len(diff),n_obs=int(counts.sum()),asymmetric_minus_symmetric_joint_score=observed,ci_low=float(lo),ci_high=float(hi),favor='asymmetric' if observed<0 else 'symmetric'))
    return pd.DataFrame(rows)

def load_returns(filename):
    if str(filename).lower().endswith('.zip'):
        with zipfile.ZipFile(filename) as archive:
            with archive.open('btc_5m_returns.csv') as f:d=pd.read_csv(f)
    else:d=pd.read_csv(filename)
    col='open_time' if 'open_time' in d else 'timestamp' if 'timestamp' in d else d.columns[0]
    if 'log_return_5m' not in d:raise ValueError('Missing log_return_5m')
    idx=pd.to_datetime(d[col],utc=True)
    return pd.Series(d.log_return_5m.to_numpy(float),index=pd.DatetimeIndex(idx)).sort_index()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--returns-csv',default='STOCHASTIC_LAB_5YEAR_RESULTS.zip')
    ap.add_argument('--train-days',default='60,120,240')
    ap.add_argument('--refit-days',type=int,default=90)
    ap.add_argument('--start-year',type=int,default=2022)
    ap.add_argument('--max-fit-obs',type=int,default=100000)
    ap.add_argument('--bootstraps',type=int,default=1500)
    ap.add_argument('--block-days',type=int,default=7)
    ap.add_argument('--output-dir',default='v09_outputs')
    a=ap.parse_args();t=time.monotonic()
    r=load_returns(a.returns_csv)
    daily,fits=run(r,[int(x.strip()) for x in a.train_days.split(',')],a.refit_days,a.start_year,a.max_fit_obs)
    output=Path(a.output_dir);output.mkdir(exist_ok=True,parents=True)
    daily.to_csv(output/'daily_scores.csv',index=False);fits.to_csv(output/'model_refits.csv',index=False)
    yearly=summarize(daily);yearly.to_csv(output/'yearly_comparison.csv',index=False)
    ci=paired_bootstrap(daily,a.bootstraps,block_days=a.block_days);ci.to_csv(output/'paired_day_bootstrap.csv',index=False)
    (output/'methodology.json').write_text(json.dumps({'source':str(a.returns_csv),'requested_training_days':[int(x) for x in a.train_days.split(',')],'refit_observed_days':a.refit_days,'maximum_training_observations_per_fit':a.max_fit_obs,'sampling':'Last N chronological returns within historical training window; actual n_fit recorded. Use max-fit-obs >= train_days * 288 to avoid truncating the specified window.','execution':'Forecast variance resets at each date and each missing interval; each forecast at t uses only returns strictly before t within current run.','innovations':'Gaussian QMLE for variance parameters, then two-stage standardized Student-t df fit. Not joint maximum likelihood.','scoring':'Joint VaR/ES score only if predicted ES > 0. Paired moving-block bootstrap (default 7 observed days), exploratory; no formal guarantee under long-run dependence.','limits':'Model-selection bias, tick data issues and intraday seasonality not eliminated. Sparse series segments are reset. This is risk forecasting, not trading profitability.'},indent=2))
    print('\nYEARLY SUMMARY\n',yearly.to_string(index=False),'\nPAIRED BOOTSTRAP\n',ci.to_string(index=False))
    print(f'Finished in {time.monotonic()-t:.1f}s. Outputs: {output}',flush=True)
if __name__=='__main__':main()
