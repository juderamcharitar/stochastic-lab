"""Chronological, day-disjoint walk-forward BTC/USDT tail-risk evaluation.

Does not carry GARCH states across missing days; trains only on dates preceding
an evaluation day. Research-only risk estimates, not trading recommendations.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, t as student_t

try:
    from src.research import load_prices, contiguous_returns, kupiec_pvalue, demo_prices
except ModuleNotFoundError:
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    from src.research import load_prices, contiguous_returns, kupiec_pvalue, demo_prices


def segment_by_day(returns):
    """Never permit a segment to cross an unobserved calendar day."""
    return [np.asarray(group.values, dtype=float) for _,group in returns.groupby(returns.index.date) if len(group)>0]


def garch_fit_segmented(segments):
    """Gaussian QMLE; per-day unconditional variance resets, no gap-spanning recurrence."""
    pooled=np.concatenate(segments)
    mu=float(np.mean(pooled)); scale=max(float(np.std(pooled,ddof=1)),1e-10)
    standardized=[(day-mu)/scale for day in segments]
    def loss(par):
        a,b=par
        if a<=0 or b<=0 or a+b>=.995: return 1e11
        w=1-a-b
        acc=0.
        for day in standardized:
            v=1.
            for r in day:
                acc+=np.log(max(v,1e-10))+r*r/max(v,1e-10)
                v=w+a*r*r+b*v
        return .5*acc
    res=minimize(loss,[.08,.86],method='L-BFGS-B',bounds=[(.001,.4),(.01,.99)],options={'maxiter':90})
    a,b=map(float,res.x)
    if a+b>=.995: a,b=.08,.86
    return {'mu':mu,'scale':scale,'alpha':a,'beta':b,'omega':1-a-b,'optimizer_success':bool(res.success)}


def one_day_garch_variances(day, fit):
    """One-step forecasts using only preceding observations on this day."""
    a,b,w=fit['alpha'],fit['beta'],fit['omega']
    mu,s=fit['mu'],fit['scale']
    v=1.
    var=[]
    for r in day:
        var.append(max(v,1e-10)*s*s)
        z=(r-mu)/s
        v=w+a*z*z+b*v
    return np.array(var)


def standardized_residuals(train_segments, fit):
    parts=[]
    for day in train_segments:
        v=one_day_garch_variances(day,fit)
        parts.append((day-fit['mu'])/np.sqrt(v))
    return np.concatenate(parts)


def fit_t_df(z):
    """Student-t innovation df via likelihood, standardized to variance one."""
    z=np.asarray(z,dtype=float)
    def obj(df):
        s=np.sqrt((df-2)/df)
        return -float(np.sum(student_t.logpdf(z/s,df)-np.log(s)))
    result=minimize_scalar(obj,bounds=(2.2,50.0),method='bounded',options={'xatol':.05})
    return float(result.x)


def block_resample_by_day(train_segments, draws, rng):
    """One-step empirical baseline; resample exclusively from observed training days."""
    pool=np.concatenate(train_segments)
    return pool[rng.integers(0,len(pool),size=draws)]


def var_forecasts(train_segments, test_day, *, seed=42):
    train=np.concatenate(train_segments)
    mu=float(train.mean()); sd=float(train.std(ddof=1));q=.01
    fit=garch_fit_segmented(train_segments)
    variances=one_day_garch_variances(test_day,fit)
    df=fit_t_df(standardized_residuals(train_segments,fit))
    rng=np.random.default_rng(seed)
    emp=np.quantile(block_resample_by_day(train_segments,100_000,rng),q)
    tcrit=student_t.ppf(q,df)*np.sqrt((df-2)/df)
    forecasts={
        'Gaussian IID':np.full(len(test_day),-(mu+sd*norm.ppf(q))),
        'Empirical Quantile':np.full(len(test_day),-emp),
        'GARCH Gaussian':-(fit['mu']+np.sqrt(variances)*norm.ppf(q)),
        'GARCH Student-t':-(fit['mu']+np.sqrt(variances)*tcrit),
    }
    return forecasts,{'garch_fit':fit,'student_t_df':df}


def quantile_score(y,qhat,q=.01):
    """Pinball scoring for left-tail RETURN quantile, lower is better."""
    return np.where(y<qhat,q-1,q)*(y-qhat)


def run_walkforward(returns, min_train_days=5, seed=42):
    grouped=[(str(day),np.asarray(g.values,dtype=float)) for day,g in returns.groupby(returns.index.date)]
    if len(grouped)<min_train_days+2:
        raise ValueError(f'Need at least {min_train_days+2} distinct observation days')
    records=[]; fits=[]
    for i in range(min_train_days,len(grouped)):
        date,test=grouped[i]
        prior=[arr for _,arr in grouped[:i]]
        if len(test)<30: continue
        forecasts,diag=var_forecasts(prior,test,seed=seed+i)
        fits.append({'date':date,'train_days':i,'test_observations':len(test),'student_t_df':diag['student_t_df'],**diag['garch_fit']})
        for name,var in forecasts.items():
            loss=-test
            breach=loss>var
            pinball=quantile_score(test,-var,.01)
            for j,(l,v,b,p) in enumerate(zip(loss,var,breach,pinball)):
                records.append({'date':date,'observation':j,'model':name,'loss':float(l),'VaR99':float(v),'breach':int(b),'pinball':float(p)})
    detail=pd.DataFrame(records)
    if detail.empty:raise ValueError('No eligible evaluation days')
    summary=[]
    for model,group in detail.groupby('model'):
        k=int(group.breach.sum());n=len(group)
        summary.append({'model':model,'n':n,'breaches':k,'breach_rate':k/n,'target':.01,'mean_pinball':float(group.pinball.mean()),'avg_VaR99':float(group.VaR99.mean()),'kupiec_pvalue_exploratory':float(kupiec_pvalue(group.breach.values,.01))})
    return detail,pd.DataFrame(summary).sort_values('mean_pinball'),pd.DataFrame(fits)


def plot_backtest(detail,summary,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    for name,g in detail.groupby('model'):
        perday=g.groupby('date').agg(rate=('breach','mean'))
        ax.plot(perday.index,perday.rate.values*100,marker='.',label=name)
    ax.axhline(1,ls='--',color='gray',label='Nominal 1%')
    ax.set(ylabel='Exceedances (%)',xlabel='Evaluation day',title='Walk-forward daily 99% VaR exceedance rates')
    ax.tick_params(axis='x',rotation=35);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'walkforward_coverage.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    ax.barh(summary.model,summary.mean_pinball*1e4)
    ax.set(xlabel='Mean 1% quantile loss (×10⁴; lower better)',title='Out-of-sample pinball scores')
    fig.tight_layout();fig.savefig(out/'walkforward_scores.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='Segment-safe, chronological walk-forward tail risk research')
    ap.add_argument('--data-dir',default='data');ap.add_argument('--output-dir',default='outputs')
    ap.add_argument('--demo',action='store_true');ap.add_argument('--min-train-days',type=int,default=5)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--max-files',type=int)
    args=ap.parse_args()
    if args.demo:
        prices=demo_prices(seed=args.seed,days=9)
        source='SYNTHETIC DEMO - NOT REAL DATA'
    else:
        prices,_=load_prices(Path(args.data_dir),args.max_files)
        source='HISTORICAL BINANCE ZIP DATA'
    returns=contiguous_returns(prices,5)
    detail,summary,fits=run_walkforward(returns,args.min_train_days,args.seed)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    detail.to_csv(out/'walkforward_observations.csv',index=False)
    summary.to_csv(out/'walkforward_summary.csv',index=False)
    fits.to_csv(out/'walkforward_fits.csv',index=False)
    plot_backtest(detail,summary,out)
    manifest={'source':source,'modeling':'rolling expanding train by observed day; predict next observed day; reset conditional GARCH variance at each day boundary','n_days':len(set(returns.index.date)),'min_train_days':args.min_train_days,'evaluation_days':int(fits.shape[0]),'n_observations':int(summary.n.iloc[0]),'limitations':['Sparse days are not equivalent to a continuous time series','Only 16 observed days in initial archive: high model uncertainty','Student-t estimated from fitted residuals, not fully joint maximum likelihood','Quantile coverage is not sufficient to establish full tail-risk calibration','No overlapping test day used to refit its own forecast','Conditional state resets every observed day; no overnight propagation','Daily dependence and multiple comparisons limit p-value interpretation','Empirical quantile is a baseline, not multi-step bootstrap simulation']}
    (out/'walkforward_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary.to_string(index=False));print('Wrote',out.resolve())

if __name__=='__main__':main()
