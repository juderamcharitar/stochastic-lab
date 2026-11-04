"""Reproducible, explanatory Monte Carlo tail-risk research. Not investment advice."""
from __future__ import annotations
import argparse, json, math, re, zipfile
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm, chi2, t as student_t
import numpy as np
import pandas as pd

ZIP_RE = re.compile(r'BTCUSDT-(?:trades|aggTrades|1m|5m)-(\d{4}-\d{2}-\d{2})\.zip$', re.I)

def timestamp_utc(a):
    v = pd.to_numeric(a, errors='coerce')
    median = v.dropna().median()
    unit = 'us' if median > 1e14 else 'ms'
    return pd.to_datetime(v, unit=unit, utc=True, errors='coerce')

def read_zip_minutes(path: Path, chunk_size=250_000):
    """Read historical Binance spot trade/aggTrade ZIP (no header or with header) or 1m kline ZIP."""
    p = Path(path)
    kind = 'aggTrades' if 'aggtrades' in p.name.lower() else 'trades'
    with zipfile.ZipFile(p) as z:
        files = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if len(files) != 1: raise ValueError(f'{p.name}: expected exactly one CSV')
        with z.open(files[0]) as f:
            first = f.readline().decode('utf-8-sig').strip().split(',')
        header = 0 if not first[0].strip('"').lstrip('-').isdigit() else None
        kline = ('-1m-' in p.name.lower()) or ('-5m-' in p.name.lower())
        if kline:
            df = pd.read_csv(z.open(files[0]), header=header, usecols=[0,4], names=['time','price'] if header is None else None)
            if header is not None: df.columns=['time','price']
            dt=timestamp_utc(df['time']); val=pd.to_numeric(df['price'],errors='coerce')
            s=pd.Series(val.to_numpy(),index=dt).dropna().sort_index()
            return s.groupby(s.index.floor('min')).last()
        # trades 7 columns (tradeId,price,qty,quoteQty,time,buyerMaker,bestMatch)
        # aggTrades 8 columns (aggTradeId,price,qty,firstId,lastId,time,buyerMaker,bestMatch)
        time_col=5 if kind=='aggTrades' else 4
        outputs=[]
        f=z.open(files[0]);
        for chunk in pd.read_csv(f,header=header,usecols=[1,time_col],chunksize=chunk_size, low_memory=False):
            if chunk.shape[1]!=2: raise ValueError(f'{p}: bad columns')
            # usecols may be returned ascending, price column then time column
            rawt=pd.to_numeric(chunk.iloc[:,1],errors='coerce').to_numpy(dtype='float64')
            rawp=pd.to_numeric(chunk.iloc[:,0],errors='coerce').to_numpy(dtype='float64')
            valid=np.isfinite(rawt)&np.isfinite(rawp)&(rawp>0)
            if not np.any(valid): continue
            rawt=rawt[valid].astype('int64');rawp=rawp[valid]
            divisor=60_000_000 if np.median(rawt)>1e14 else 60_000
            c=pd.DataFrame({'minute':rawt//divisor,'timestamp':rawt,'price':rawp})
            c=c.sort_values('timestamp').groupby('minute',sort=False).tail(1)
            outputs.append(c)
        if not outputs: raise ValueError(f'{p}: empty')
        allc=pd.concat(outputs,ignore_index=True)
        allc=allc.sort_values('timestamp').groupby('minute',sort=True).tail(1).sort_values('minute')
        return pd.Series(allc.price.to_numpy(), index=pd.to_datetime(allc.minute.to_numpy(dtype='int64'),unit='m',utc=True), name='price')

def load_prices(folder:Path, max_files=None):
    paths=sorted([p for p in Path(folder).rglob('*.zip') if p.name.startswith('BTCUSDT-') and ('trades-' in p.name.lower() or 'aggtrades-' in p.name.lower() or '-1m-' in p.name.lower())])
    if max_files: paths=paths[:max_files]
    if not paths: raise FileNotFoundError(f'No BTCUSDT trade ZIPs in {folder}')
    series=[]; manifest=[]
    for p in paths:
        s=read_zip_minutes(p)
        if s.empty: continue
        series.append(s)
        manifest.append({'archive':p.name,'minutes':len(s),'first':str(s.index.min()),'last':str(s.index.max())})
        print(f'  loaded {p.name}: {len(s)} populated minutes',flush=True)
    prices=pd.concat(series).sort_index()
    duplicated=prices.index.duplicated(keep='last')
    prices=prices[~duplicated]
    return prices,manifest

def contiguous_returns(prices: pd.Series, horizon_minutes=5):
    """Returns from t-5min to t only when exactly five minutes apart; never bridge gaps."""
    sampled=prices[prices.index.minute % horizon_minutes == 0]
    elapsed=sampled.index.to_series().diff().dt.total_seconds()
    returns=np.log(sampled/sampled.shift(1))
    returns=returns[elapsed.eq(horizon_minutes*60)].dropna()
    if len(returns)<100: raise ValueError('Insufficient consecutive bars; need at least 100 returns')
    return returns

def normal_params(x): return float(np.mean(x)),max(float(np.std(x,ddof=1)),1e-8)

def mixture_fit(x):
    """Two-Gaussian mixture, shared mean, fitted by maximum likelihood with ordered sigmas.
    This represents a phenomenological jump-like mixture, not identified causal news jumps.
    """
    mu,s=normal_params(x)
    # scale to percentage points for stable optimizer
    xp=np.asarray(x)*100
    m,sd=normal_params(xp)
    def nll(z):
        mean,logsd,logratio,logit=z
        low=np.exp(logsd); high=low*(1+np.exp(logratio)); w=1/(1+np.exp(-logit))
        a=np.logaddexp(np.log1p(-w)+norm.logpdf(xp,mean,low),np.log(w)+norm.logpdf(xp,mean,high))
        return -float(np.sum(a)) if np.all(np.isfinite(a)) else 1e100
    fit=minimize(nll,[m,np.log(sd*.65),np.log(2),-2.2],method='L-BFGS-B',bounds=[(m-2*sd,m+2*sd),(np.log(sd*.1),np.log(sd*3)),(-3,3),(-6,0)])
    if not fit.success and not np.isfinite(fit.fun): raise RuntimeError('Mixture fit failed')
    a,b,c,d=fit.x
    return {'mu':float(a/100),'sigma_calm':float(np.exp(b)/100),'sigma_jump':float(np.exp(b)*(1+np.exp(c))/100),'jump_prob':float(1/(1+np.exp(-d))),'optimizer_success':bool(fit.success)}

def garch_variance(x, omega,alpha,beta,init=None):
    """Gaussian GARCH(1,1); x in percent, x is demeaned; h_t conditioned on x_{t-1}."""
    x=np.asarray(x)
    h=np.empty_like(x,dtype=float)
    v=float(np.var(x)) if init is None else float(init)
    for i in range(len(x)):
        h[i]=max(v,1e-10)
        v=omega+alpha*x[i]**2+beta*v
    return h,v

def garch_fit(x):
    # normalized to percentage points, standardized by training sample sigma to bound typical params
    mu,s=normal_params(x); z=(np.asarray(x)-mu)/s
    def neg_ll(theta):
        a,b=theta
        if a<0 or b<0 or a+b>=.999: return 1e10
        omega=1-a-b
        h,_=garch_variance(z,omega,a,b,1.0)
        return float(.5*np.sum(np.log(h)+z*z/h))
    out=minimize(neg_ll,[.08,.85],method='L-BFGS-B',bounds=[(.0001,.5),(.0001,.998)],options={'maxiter':250})
    a,b=out.x
    if a+b>=.999: a,b=.08,.85
    return {'mu':mu,'scale':s,'alpha':float(a),'beta':float(b),'omega':float(1-a-b),'optimizer_success':bool(out.success)}

def garch_forecast_vol(train,test,fit):
    mu,s=fit['mu'],fit['scale']
    a,b,w=fit['alpha'],fit['beta'],fit['omega']
    _,v=garch_variance((np.asarray(train)-mu)/s,w,a,b,1.)
    out=[]
    for r in np.asarray(test):
        out.append(s*np.sqrt(max(v,1e-10)))
        v=w+a*((r-mu)/s)**2+b*v
    return np.asarray(out)

def gaussian_var_es(mu,sigma,q=.01):
    z=norm.ppf(q)
    return -mu-sigma*z, -mu+sigma*norm.pdf(z)/q

def empirical_var_es(draws,q=.01):
    threshold=np.quantile(draws,q)
    return -float(threshold),-float(np.mean(draws[draws<=threshold]))

def kupiec_pvalue(breaches,q):
    n=len(breaches); k=int(np.sum(breaches)); ph=k/n
    eps=1e-12
    ll0=k*np.log(q)+(n-k)*np.log(1-q)
    ll1=k*np.log(max(ph,eps))+(n-k)*np.log(max(1-ph,eps))
    return float(chi2.sf(max(0,2*(ll1-ll0)),1))

def bootstrap_paths(x,steps,npaths,block,rng):
    x=np.asarray(x); out=np.empty((npaths,steps))
    for i in range(npaths):
        arr=[]
        while len(arr)<steps:
            st=int(rng.integers(0,max(1,len(x)-block+1)))
            arr.extend(x[st:st+block].tolist())
        out[i,:]=arr[:steps]
    return out

def simulate_paths(train,mixture,garch,steps=48,npaths=3000,seed=42):
    rng=np.random.default_rng(seed)
    mu,s=normal_params(train)
    gaussian=rng.normal(mu,s,size=(npaths,steps))
    m=mixture
    jump=rng.random((npaths,steps))<m['jump_prob']
    sigma=np.where(jump,m['sigma_jump'],m['sigma_calm'])
    mix=rng.normal(m['mu'],sigma)
    boot=bootstrap_paths(train,steps,npaths,block=12,rng=rng)
    z=(np.asarray(train)-garch['mu'])/garch['scale']
    _,v=garch_variance(z,garch['omega'],garch['alpha'],garch['beta'])
    g=np.empty((npaths,steps)); v=np.full(npaths,v)
    for j in range(steps):
        shock=rng.normal(size=npaths)
        g[:,j]=garch['mu']+garch['scale']*np.sqrt(v)*shock
        v=garch['omega']+garch['alpha']*(np.sqrt(v)*shock)**2+garch['beta']*v
    return {'Gaussian IID':gaussian,'Gaussian Mixture':mix,'GARCH(1,1)':g,'Block Bootstrap':boot}

def evaluate(train,test,seed=42,q=.01):
    mu,s=normal_params(train)
    mixture=mixture_fit(train);garch=garch_fit(train)
    rng=np.random.default_rng(seed)
    n_draws=200_000
    jump=rng.random(n_draws)<mixture['jump_prob']
    mixture_draws=rng.normal(mixture['mu'],np.where(jump,mixture['sigma_jump'],mixture['sigma_calm']))
    bs_draws=np.asarray(train)[rng.integers(len(train),size=n_draws)]
    forecasts={
      'Gaussian IID':np.full(len(test),gaussian_var_es(mu,s,q)[0]),
      'Gaussian Mixture':np.full(len(test),empirical_var_es(mixture_draws,q)[0]),
      'GARCH(1,1)':-garch['mu']-norm.ppf(q)*garch_forecast_vol(train,test,garch),
      'Empirical Bootstrap':np.full(len(test),empirical_var_es(bs_draws,q)[0]),
    }
    rows=[]
    for name, var in forecasts.items():
        breach=(-np.asarray(test)>var)
        rows.append({'model':name,'n_test':len(test),'breaches':int(breach.sum()),'breach_rate':float(breach.mean()),'target_rate':q,'kupiec_pvalue':kupiec_pvalue(breach,q),'mean_VaR_pct':float(np.mean(var)*100)})
    return pd.DataFrame(rows),mixture,garch,forecasts

def plots(out,returns,train,test,paths,forecasts):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(returns.index,returns.values*100,lw=.5)
    ax.axvline(test.index.min(),ls='--',lw=1,label='Holdout begins')
    ax.set(title='Observed BTCUSDT 5-minute log returns (nonconsecutive days excluded)',ylabel='Log return (%)',xlabel='UTC time');ax.legend();fig.tight_layout();fig.savefig(out/'observed_returns.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for name,arr in paths.items():
        end=np.exp(np.sum(arr,axis=1))*100
        ax.hist(end,bins=75,density=True,histtype='step',lw=1.5,label=name)
    ax.set(title='Simulated 4-hour terminal value of $100 (training calibration)',xlabel='Ending value ($)',ylabel='Density');ax.legend();fig.tight_layout();fig.savefig(out/'terminal_distributions.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for name,arr in paths.items():
        path=np.exp(np.cumsum(arr,axis=1))*100
        low,med,hi=np.quantile(path,[.05,.5,.95],axis=0)
        ax.plot(np.arange(1,path.shape[1]+1)*5,med,label=name)
        ax.fill_between(np.arange(1,path.shape[1]+1)*5,low,hi,alpha=.08)
    ax.set(title='Monte Carlo conditional simulation (median, 5–95% interval)',xlabel='Minutes into simulated future',ylabel='$100 starting value');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'scenario_paths.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    tail=-test.to_numpy()*100
    ax.plot(test.index,tail,lw=.7,label='Realized loss (%)')
    for name,var in forecasts.items():
        ax.plot(test.index,var*100,lw=.75,alpha=.8,label=f'{name} 99% VaR')
    ax.set(title='Untouched holdout: realized 5-minute losses vs model 99% VaR',ylabel='Loss / VaR (%)',xlabel='UTC');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'var_backtest.png',dpi=150);plt.close(fig)

def demo_prices(seed=42,days=24):
    rng=np.random.default_rng(seed)
    ix=pd.date_range('2026-01-01',periods=days*1440,freq='min',tz='UTC')
    z=rng.normal(size=len(ix)); v=np.empty(len(ix));v[0]=.0003**2
    for i in range(1,len(ix)): v[i]=.03*.0003**2+.07*(np.sqrt(v[i-1])*z[i-1])**2+.9*v[i-1]
    r=np.sqrt(v)*z
    r+=rng.normal(0,.004,len(ix))*(rng.random(len(ix))<.0007)
    return pd.Series(75000*np.exp(np.cumsum(r)),index=ix,name='price')

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--data-dir',default='data');a.add_argument('--output-dir',default='outputs');a.add_argument('--demo',action='store_true')
    a.add_argument('--max-files',type=int);a.add_argument('--paths',type=int,default=3000);a.add_argument('--seed',type=int,default=42)
    args=a.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if args.demo:
        prices=demo_prices(seed=args.seed);manifest=[{'source':'SYNTHETIC GARCH with rare shocks; not observed market data'}]
        source='SYNTHETIC DEMONSTRATION ONLY'
    else:
        prices,manifest=load_prices(Path(args.data_dir),args.max_files);source='BINANCE HISTORICAL SPOT CSV ZIP ARCHIVES'
    returns=contiguous_returns(prices,5)
    # STRICT holdout: split by calendar day (so no day is partly train/test).
    dates=sorted(set(returns.index.date))
    cut=max(1,int(len(dates)*.7))
    train=returns[np.isin(returns.index.date,dates[:cut])]
    test=returns[np.isin(returns.index.date,dates[cut:])]
    if len(train)<100 or len(test)<30: raise ValueError('Need more observation days to form train/test split')
    table,mixture,garch,forecasts=evaluate(train,test,args.seed)
    paths=simulate_paths(train,mixture,garch,48,args.paths,args.seed)
    plots(out,returns,train,test,paths,forecasts)
    table.to_csv(out/'backtest.csv',index=False)
    summary={'source':source,'days':len(dates),'days_train':len(dates[:cut]),'days_test':len(dates[cut:]),'returns_train':len(train),'returns_test':len(test),'start_utc':str(prices.index.min()),'end_utc':str(prices.index.max()),'observed_minutes':len(prices),'mixture_fit':mixture,'garch_fit':garch,'archives':manifest,'caveats':['Nonconsecutive days are not bridged for returns','Five-minute returns are formed from populated close-minute prices','Model fit is TRAIN ONLY; holdout days never used for parameter fitting','Coverage test assumptions require caution due to temporal dependence','Simulation is not a trading strategy, causal attribution or forward guarantee','Mixture wide component is not proof of causal price jumps','Multiple model comparison and single holdout add selection uncertainty']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('\nSOURCE:',source,'\nObserved minutes:',len(prices),'5m returns:',len(returns),'Train:',len(train),'Test:',len(test))
    print(table.to_string(index=False))
    print('\nWrote files to:',out.resolve())
if __name__=='__main__': main()
