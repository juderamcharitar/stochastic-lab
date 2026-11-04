"""V1.0: Monte Carlo scenario modeling with a strictly held-out 2025 reality check.

Calibrate models only on 2021–2024 UTC 5-minute returns; evaluate calendar-complete
2025 days. Synthetic paths are model scenarios, not future return forecasts.
Run: python -m src.market_sim_v10 --returns-csv btc_5m_returns.csv
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import norm
from src.robustness_v09 import fit_model

STEPS=288

def daily_panels(returns:pd.Series, steps:int=STEPS):
    """Require complete, uninterrupted UTC 5-minute windows; avoid gap returns."""
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError('timestamps must be strictly increasing and unique')
    out=[]; dates=[]
    for day,g in returns.groupby(returns.index.floor('D')):
        ix=g.index
        if len(g)!=steps or not np.all(np.diff(ix.asi8)==pd.Timedelta(minutes=5).value):
            continue
        # This return series has 00:05 ... 23:55 and 00:00 next day as a standard
        # full 288 5-min day under the dataset's original candle-close timestamp
        # convention. Preserve exact count and contiguity, no infill.
        out.append(g.to_numpy(dtype=float));dates.append(day)
    return pd.DatetimeIndex(dates), np.stack(out) if out else np.empty((0,steps))

def summarize_paths(paths):
    paths=np.asarray(paths,float)
    end=paths.sum(axis=1)
    running=np.c_[np.zeros(len(paths)),np.cumsum(paths,axis=1)]
    drawdown=np.min(running-np.maximum.accumulate(running,axis=1),axis=1)
    rv=np.sqrt((paths**2).sum(axis=1))
    return {'daily_logreturn':end,'max_log_drawdown':drawdown,'realized_vol':rv}

def calibrate(training, max_fit_obs=70000):
    r=training.to_numpy(float)
    mu=float(np.mean(r));sigma=float(np.std(r,ddof=1))
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('Invalid training volatility')
    # Two-stage Student-t fit on standardized iid returns.
    from scipy.optimize import minimize_scalar
    z=(r-mu)/sigma
    def negll(df):
        scale=np.sqrt((df-2)/df)
        return float(-np.mean(student_t.logpdf(z/scale,df)-np.log(scale)))
    df_iid=float(minimize_scalar(negll,bounds=(2.2,60),method='bounded',options={'xatol':.05}).x)
    sym=fit_model(training,False,max_fit_obs=max_fit_obs,maxiter=50)
    asym=fit_model(training,True,max_fit_obs=max_fit_obs,maxiter=50)
    # Exploratory jump-diffusion calibration: identify large standardized 5-minute
    # shocks; do not treat inferred jumps as structurally identified events.
    jump_mask=np.abs(z)>4
    jump_z=z[jump_mask]
    jump_rate=float(jump_mask.mean())
    jump_mean=float(jump_z.mean()) if len(jump_z) else 0.0
    jump_sd=float(jump_z.std(ddof=1)) if len(jump_z)>1 else 0.0
    return {'mean':mu,'std':sigma,'student_df':df_iid,
            'jump_probability':jump_rate,'jump_mean_z':jump_mean,'jump_sd_z':jump_sd,
            'symmetric':sym,'asymmetric':asym,
            'fit_dates':[str(training.index[0]),str(training.index[-1])], 'n_training_obs':len(training)}

def generate(model,params,n_paths=10000,steps=STEPS,seed=42,block_source=None):
    if n_paths<=0 or steps<=0:raise ValueError('positive paths and horizon required')
    rng=np.random.default_rng(seed)
    if model=='Gaussian IID':return rng.normal(params['mean'],params['std'],size=(n_paths,steps))
    if model=='Heuristic jump-diffusion':
        # Illustrative compound Bernoulli jump mixture, calibrated with 4-sigma
        # extreme returns. No deconvolution; variance can be double-counted.
        base=rng.normal(params['mean'],params['std'],size=(n_paths,steps))
        jump=rng.random((n_paths,steps))<params['jump_probability']
        if jump.any():
            impulse=rng.normal(params['jump_mean_z'],params['jump_sd_z'],size=int(jump.sum()))
            base[jump]+=params['std']*impulse
        return base
    if model=='Student-t IID':
        df=params['student_df'];return params['mean']+params['std']*np.sqrt((df-2)/df)*rng.standard_t(df,size=(n_paths,steps))
    if model=='Historical day bootstrap':
        if block_source is None or len(block_source)==0:raise ValueError('historical days required')
        if block_source.shape[1]!=steps:raise ValueError('bootstrap horizon mismatch')
        return block_source[rng.integers(0,len(block_source),size=n_paths)].copy()
    if model not in ('Student-t GARCH','Student-t GJR-GARCH'):raise ValueError(model)
    p=params['symmetric'] if model=='Student-t GARCH' else params['asymmetric']
    mu,sig,df=p['mu'],p['scale'],p['df']
    a,b,g=p['alpha'],p['beta'],p['gamma']
    omega=1-a-b-g/2
    if omega<=0:raise ValueError('nonstationary volatility parameters')
    h=np.ones(n_paths)
    result=np.empty((n_paths,steps),dtype=np.float32)
    for t in range(steps):
        z=rng.standard_t(df,size=n_paths)*np.sqrt((df-2)/df)
        result[:,t]=mu+sig*np.sqrt(h)*z
        h=np.maximum(omega+(a+g*(z<0))*h*z*z+b*h,1e-10)
    return result

def quantile_score(observed, q_value, alpha):
    # lower-tail quantile of daily log return; lower is better
    y=np.asarray(observed,float)
    return float(np.mean((alpha-(y<q_value))*(y-q_value)))

def crps_mc(observed, sims, seed=1, sampled_observations=2000):
    """Monte Carlo CRPS from empirical sample via exact sample-vs-sample expectation.
    Uses up to sampled_observations draws; not an inferential p-value.
    """
    rng=np.random.default_rng(seed)
    z=np.asarray(sims,float)
    if len(z)>sampled_observations:z=rng.choice(z,sampled_observations,replace=False)
    z.sort()
    # E|X-y| evaluated by searchsorted, and E|X-X'| via sorted sum.
    idx=np.searchsorted(z,observed)
    cs=np.r_[0,z.cumsum()];n=len(z)
    left=observed*idx-cs[idx]
    right=(cs[-1]-cs[idx])-observed*(n-idx)
    e1=np.mean((left+right)/n)
    e2=2*np.sum((2*np.arange(n)-n+1)*z)/(n*n)
    return float(e1-0.5*e2)

def evaluate(simulated,observed,model):
    obs=summarize_paths(observed)
    sim=summarize_paths(simulated)
    loss=-obs['daily_logreturn'];model_loss=-sim['daily_logreturn']
    v=float(np.quantile(model_loss,.99));es=float(np.mean(model_loss[model_loss>=v]))
    breaches=loss>v
    # Simple descriptive ES diagnostic for few daily breaches, not formal calibration.
    tail_realized=float(loss[breaches].mean()) if breaches.any() else np.nan
    d={'model':model,'historical_days':len(loss),'scenario_paths':len(simulated),'var99_logreturn_loss':v,
       'es99_logreturn_loss':es,'observed_var_breaches':int(breaches.sum()),
       'observed_breach_rate':float(breaches.mean()),'observed_mean_loss_on_breaches':tail_realized,
       'daily_return_crps':crps_mc(obs['daily_logreturn'],sim['daily_logreturn']),
       'daily_return_pinball_1pct':quantile_score(obs['daily_logreturn'],float(np.quantile(sim['daily_logreturn'],.01)),.01),
       'observed_median_daily_rv':float(np.median(obs['realized_vol'])),
       'simulated_median_daily_rv':float(np.median(sim['realized_vol'])),
       'observed_median_max_drawdown':float(np.median(obs['max_log_drawdown'])),
       'simulated_median_max_drawdown':float(np.median(sim['max_log_drawdown'])),
       'simulated_1pct_daily_return':float(np.quantile(sim['daily_logreturn'],.01)),
       'simulated_99pct_daily_return':float(np.quantile(sim['daily_logreturn'],.99))}
    return d

def run(r,paths=10000,seed=2026,out_dir='v10_outputs',max_fit_obs=70000):
    if r.index.tz is None:raise ValueError('UTC timestamps required')
    training=r[r.index.year<=2024]
    test=r[r.index.year==2025]
    if len(training)<10000 or len(test)<10000:raise ValueError('Need 2021–2024 training and 2025 holdout')
    train_dates,train_days=daily_panels(training)
    holdout_dates,holdout=daily_panels(test)
    if len(train_days)<100 or len(holdout)<100:raise ValueError('Insufficient complete days')
    print(f'Calibrating on {len(training):,} 2021–2024 observations; holding out {len(holdout)} complete 2025 days',flush=True)
    pars=calibrate(training,max_fit_obs=max_fit_obs)
    out=Path(out_dir);out.mkdir(exist_ok=True,parents=True)
    # Save holdout days for transparent observed-vs-model comparison.
    pd.DataFrame({'date':holdout_dates.astype(str),
        'daily_logreturn':holdout.sum(axis=1),
        'realized_vol':np.sqrt((holdout**2).sum(axis=1))}).to_csv(out/'holdout_2025_days.csv',index=False)
    models=['Gaussian IID','Student-t IID','Student-t GARCH','Student-t GJR-GARCH','Historical day bootstrap','Heuristic jump-diffusion']
    results=[];sample=[]
    for j,model in enumerate(models):
        sim=generate(model,pars,paths,STEPS,seed+j*97,train_days)
        result=evaluate(sim,holdout,model)
        results.append(result)
        sample.append(pd.DataFrame({'model':model,'path':np.arange(min(250,paths)),
            'daily_logreturn':sim[:250].sum(axis=1),
            'max_log_drawdown':summarize_paths(sim[:250])['max_log_drawdown']}))
        print(f'{model:26s} VaR breaches: {result["observed_var_breaches"]}/{result["historical_days"]}; CRPS: {result["daily_return_crps"]:.6g}',flush=True)
    pd.DataFrame(results).to_csv(out/'heldout_model_comparison.csv',index=False)
    pd.concat(sample,ignore_index=True).to_csv(out/'scenario_samples.csv',index=False)
    # Vary horizon and volatility multiplier on the *same* scenario paths.
    # The risk surface is a sensitivity analysis, not model calibration.
    risk_grid=[]
    for mult in (.6,.8,1.,1.2,1.5,2.):
        z=generate('Student-t GJR-GARCH',pars,4000,STEPS*7,seed+707)
        for days in (1,2,3,5,7):
            loss=-z[:,:STEPS*days].sum(axis=1)*mult
            q=np.quantile(loss,.99)
            risk_grid.append(dict(horizon_days=days,volatility_multiplier=mult,var99=float(q),es99=float(np.mean(loss[loss>=q]))))
    pd.DataFrame(risk_grid).to_csv(out/'risk_surface.csv',index=False)
    # Strict convergence study for scenario-tail estimation; independent fresh seeds.
    conv=[]
    for n in (500,2000,10000):
        for rep in range(12):
            sim=generate('Student-t GJR-GARCH',pars,n,STEPS,seed+10000+n*10+rep)
            losses=-sim.sum(axis=1)
            q=np.quantile(losses,.99)
            conv.append({'paths':n,'replicate':rep,'var99':float(q),'es99':float(np.mean(losses[losses>=q]))})
    pd.DataFrame(conv).to_csv(out/'mc_convergence.csv',index=False)
    # Parameters are locked prior to validation. Confidence of 2025 coverage is limited.
    (out/'methodology.json').write_text(json.dumps({'design':'Frozen calibration on 2021–2024, evaluation on complete 2025 UTC days only',
       'models':models,'five_minute_steps_per_path':STEPS,'paths_per_model':paths,
       'training_dates':pars['fit_dates'],'n_training_returns':pars['n_training_obs'],
       'n_training_complete_days':len(train_days),'n_holdout_complete_days':len(holdout),
       'seed':seed,'model_parameters':pars,'notes':["Date-based holdout is never used for fitting or choosing models.",
       'Simulation uses unconditional daily h=1 reset, rather than adaptive 2025 rolling volatility state.',
       'Historical bootstrap resamples complete training days, retaining within-day dependence only.',
       'Heuristic jump mixture uses >4 standard-deviation events; not identified jump-diffusion or variance-neutral calibration.',
       'The 3D risk surface scales GJR simulated returns; it is a scenario sensitivity map, not an empirical robustness landscape.',
       'Daily VaR coverage has only roughly 365 trials; 1% tail counts are sparse.',
       'Student-t GARCH variance parameters use Gaussian QMLE followed by separate Student-t df fitting; not full Student-t MLE.',
       'Return series is based on 5-minute candle returns, not 2025 live or order-level data.',
       'This is model-risk research, not a prospective trading or calibrated price prediction.']},indent=2,default=float))
    return pd.DataFrame(results)

def main():
    p=argparse.ArgumentParser();p.add_argument('--returns-csv',default='btc_5m_returns.csv');p.add_argument('--paths',type=int,default=10000);p.add_argument('--seed',type=int,default=2026);p.add_argument('--out-dir',default='v10_outputs');p.add_argument('--fit-obs',type=int,default=70000)
    a=p.parse_args()
    from src.robustness_v09 import load_returns
    print(run(load_returns(a.returns_csv),a.paths,a.seed,a.out_dir,a.fit_obs).to_string(index=False))
if __name__=='__main__':main()
