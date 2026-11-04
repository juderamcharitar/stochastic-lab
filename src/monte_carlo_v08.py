"""V0.8 importance sampling for rare Gaussian tail loss; analytic benchmark.
The shifted-normal likelihood ratio is exact; only synthetic Gaussian reference.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

def true_tail(q=.01,sigma=.02):
    z=norm.ppf(1-q)
    return sigma*z,sigma*norm.pdf(z)/q

def estimate(n,seed,method='iid',q=.01,sigma=.02,shift=2.35):
    if n<100:raise ValueError('n too small')
    rng=np.random.default_rng(seed);threshold,truth=true_tail(q,sigma)
    if method=='iid':
        z=rng.standard_normal(n);w=np.ones(n)
    elif method=='importance':
        # z sampled from N(shift,1), normal density ratio p/q.
        z=rng.normal(shift,1,n)
        w=np.exp(-shift*z+.5*shift**2)
    else:raise ValueError('method')
    # True-threshold unconditional ES representation: E[L 1(L>VaR)]/q
    loss=sigma*z
    estimate_es=np.mean(w*loss*(loss>threshold))/q
    return float(estimate_es)

def run(samples=(1000,4000,16000),replications=120,seed=2026,q=.01,sigma=.02):
    _,truth=true_tail(q,sigma);rows=[]
    for n in samples:
        for k in range(replications):
            # Same seed indexing scheme for reproducibility; methods use separate RNG streams.
            for m in ['iid','importance']:
                val=estimate(n,int(np.random.SeedSequence([seed,n,k,int(m=='importance')]).generate_state(1)[0]),m,q,sigma)
                rows.append({'paths':n,'replicate':k,'method':m,'estimate':val,'error':val-truth})
    raw=pd.DataFrame(rows)
    summary=raw.groupby(['paths','method']).agg(rmse=('error',lambda x:float(np.sqrt(np.mean(np.square(x))))),bias=('error','mean'),sd=('estimate','std')).reset_index()
    baseline=summary[summary.method=='iid'].set_index('paths')['sd']
    summary['variance_reduction_factor']=summary.apply(lambda x:float((baseline.loc[x.paths]/x.sd)**2),axis=1)
    return raw,summary,truth

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='v08_outputs');p.add_argument('--replications',type=int,default=120)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    raw,summary,truth=run(replications=a.replications)
    summary.to_csv(out/'mc_tail_convergence.csv',index=False);raw.to_csv(out/'mc_tail_replications.csv',index=False)
    print('Analytic Gaussian 99% ES:',truth);print(summary.to_string(index=False))
if __name__=='__main__':main()
