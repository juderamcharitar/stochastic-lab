"""Monte Carlo numerical convergence and variance reduction with known Black-Scholes benchmark.
All price models and parameters here are explicit *synthetic* assumptions.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

def bs_call(s0,k,r,sigma,t):
    if sigma<=0 or t<=0:raise ValueError('Positive sigma and horizon required')
    d1=(np.log(s0/k)+(r+.5*sigma*sigma)*t)/(sigma*np.sqrt(t))
    d2=d1-sigma*np.sqrt(t)
    return float(s0*norm.cdf(d1)-k*np.exp(-r*t)*norm.cdf(d2))

def estimate_call(n,method,rng,s0=100,k=100,r=.03,sigma=.28,t=1.0,pilot_n=2048):
    if n<4 or n%2:raise ValueError('n must be even and >=4')
    def draws(z):
        terminal=s0*np.exp((r-.5*sigma*sigma)*t+sigma*np.sqrt(t)*z)
        return np.exp(-r*t)*np.maximum(terminal-k,0),np.exp(-r*t)*terminal
    if method=='iid':
        p,_=draws(rng.standard_normal(n));return float(p.mean())
    if method=='antithetic':
        z=rng.standard_normal(n//2)
        p1,_=draws(z);p2,_=draws(-z)
        return float(((p1+p2)/2).mean())
    if method in ['control','antithetic_control']:
        # Separate independent pilot makes the control coefficient independent of main sample.
        p0,c0=draws(rng.standard_normal(pilot_n))
        beta=np.cov(p0,c0,ddof=1)[0,1]/np.var(c0,ddof=1)
        if method=='control':
            p,c=draws(rng.standard_normal(n))
        else:
            z=rng.standard_normal(n//2)
            p1,c1=draws(z);p2,c2=draws(-z)
            p=(p1+p2)/2;c=(c1+c2)/2
        return float(p.mean()-beta*(c.mean()-s0))
    raise ValueError('unknown method')

def convergence(samples=(512,2048,8192),replications=40,seed=2026,params=None):
    params=params or dict(s0=100,k=100,r=.03,sigma=.28,t=1.)
    analytic=bs_call(**params)
    methods=('iid','antithetic','control','antithetic_control')
    rows=[]
    for n in samples:
        for method in methods:
            for rep in range(replications):
                rng=np.random.default_rng(np.random.SeedSequence([seed,int(n),rep,methods.index(method)]))
                val=estimate_call(n,method,rng,**params)
                rows.append({'n_paths':n,'method':method,'replication':rep,'price':val,'error':val-analytic})
    raw=pd.DataFrame(rows)
    summary=raw.groupby(['n_paths','method']).agg(mean_price=('price','mean'),rmse=('error',lambda v:float(np.sqrt(np.mean(np.asarray(v)**2)))),replicate_sd=('price','std'),bias=('error','mean')).reset_index()
    baseline=summary[summary.method=='iid'].set_index('n_paths').replicate_sd
    summary['variance_reduction_vs_iid']=summary.apply(lambda x:float((baseline[x.n_paths]/x.replicate_sd)**2) if x.replicate_sd>0 else np.nan,axis=1)
    return raw,summary,analytic

def gaussian_tail_convergence(samples=(512,2048,8192),replications=40,seed=2026,mu=0,sigma=.02,q=.01):
    """Normal one-step loss VaR/ES: demonstrate tail convergence; no biased symmetry forced on empirical crypto data."""
    expected_var=-mu-sigma*norm.ppf(q)
    expected_es=-mu+sigma*norm.pdf(norm.ppf(q))/q
    rows=[]
    for n in samples:
        for method in ('iid','antithetic'):
            for rep in range(replications):
                rng=np.random.default_rng(np.random.SeedSequence([seed+11,int(n),rep,0 if method=='iid' else 1]))
                if method=='iid':z=rng.standard_normal(n)
                else:
                    half=rng.standard_normal(n//2);z=np.r_[half,-half]
                loss=-mu-sigma*z
                v=float(np.quantile(loss,1-q));tail=loss[loss>=v]
                es=float(tail.mean())
                rows.append({'n_paths':n,'method':method,'replication':rep,'VaR_error':v-expected_var,'ES_error':es-expected_es})
    raw=pd.DataFrame(rows)
    sm=raw.groupby(['n_paths','method']).agg(VaR_rmse=('VaR_error',lambda v:float(np.sqrt(np.mean(np.asarray(v)**2)))),ES_rmse=('ES_error',lambda v:float(np.sqrt(np.mean(np.asarray(v)**2))))).reset_index()
    return raw,sm,{'analytic_VaR':expected_var,'analytic_ES':expected_es}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',default='v07_outputs');ap.add_argument('--replications',type=int,default=40);ap.add_argument('--seed',type=int,default=2026)
    a=ap.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    raw,summary,analytic=convergence(replications=a.replications,seed=a.seed)
    raw.to_csv(out/'mc_option_replications.csv',index=False)
    summary.to_csv(out/'mc_option_convergence.csv',index=False)
    raw_t,sm_t,anal_t=gaussian_tail_convergence(replications=a.replications,seed=a.seed)
    raw_t.to_csv(out/'mc_gaussian_tail_replications.csv',index=False)
    sm_t.to_csv(out/'mc_gaussian_tail_convergence.csv',index=False)
    (out/'mc_benchmarks.json').write_text(json.dumps({'synthetic_black_scholes_call':analytic,'synthetic_gaussian_loss_tail':anal_t,'methods':['iid','antithetic','control with independent pilot','antithetic with independent pilot control'],'disclaimer':'Synthetic reference experiments; no profits/trades; real market ES backtest is separate'},indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for method,g in summary.groupby('method'):
        ax.plot(g.n_paths,g.rmse,marker='o',label=method)
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('Simulated terminal paths per replication');ax.set_ylabel('RMSE against analytic BS call price');ax.legend(fontsize=8);ax.set_title('Monte Carlo convergence and variance reduction')
    fig.tight_layout();fig.savefig(out/'mc_convergence.png',dpi=140);plt.close(fig)
    print('Analytic call:',analytic,'\n',summary.to_string(index=False));print('SAVED',out)
if __name__=='__main__':main()
