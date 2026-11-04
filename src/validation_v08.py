"""V0.8 diagnostics on saved one-step out-of-sample forecasts. No refitting.
Use: python -m src.validation_v08 --results-zip STOCHASTIC_LAB_5YEAR_RESULTS.zip
"""
import argparse,zipfile,json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import chi2

Q=.01

def joint_score(loss,var,es,q=Q):
    """FZ-style homogeneous joint VaR/ES score (loss convention, positive VaR/ES).
    Additive constants do not matter for within-sample model comparisons.
    """
    l=np.asarray(loss,float);v=np.asarray(var,float);e=np.asarray(es,float)
    if np.any(e<=0): raise ValueError('Expected Shortfall must be positive')
    return ((l-v)*(l>v)/q+v)/e+np.log(e)-1

def kupiec(b,q=Q):
    b=np.asarray(b,dtype=int);n=len(b);x=int(b.sum())
    if n==0:return np.nan,np.nan
    p=x/n;eps=1e-14
    ll0=x*np.log(q)+(n-x)*np.log1p(-q)
    ll1=x*np.log(np.clip(p,eps,1-eps))+(n-x)*np.log1p(-np.clip(p,eps,1-eps))
    lr=max(0.,2*(ll1-ll0));return lr,float(chi2.sf(lr,1))

def independence_test(b,timestamps):
    """Christoffersen Markov LR-independence; skip transitions across missing 5m candles."""
    b=np.asarray(b,dtype=int);t=pd.to_datetime(timestamps,utc=True)
    mask=np.diff(t.to_numpy(dtype='datetime64[ns]'))==np.timedelta64(5,'m')
    a=b[:-1][mask];c=b[1:][mask]
    if len(a)<10:return np.nan,np.nan,0
    counts=np.array([[np.sum((a==i)&(c==j)) for j in (0,1)] for i in (0,1)],dtype=float)
    if np.any(counts.sum(axis=1)==0):return np.nan,np.nan,len(a)
    eps=1e-12
    p=np.clip(counts[:,1]/counts.sum(axis=1),eps,1-eps)
    overall=np.clip(counts[:,1].sum()/counts.sum(),eps,1-eps)
    ll_ind=(counts[:,1]*np.log(p)+counts[:,0]*np.log1p(-p)).sum()
    ll_iid=counts[:,1].sum()*np.log(overall)+counts[:,0].sum()*np.log1p(-overall)
    lr=max(0.,2*(ll_ind-ll_iid));return lr,float(chi2.sf(lr,1)),len(a)

def load_saved(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        needed='var_es_observations.csv'
        if needed not in z.namelist():raise ValueError('Missing '+needed)
        with z.open(needed) as f:df=pd.read_csv(f,parse_dates=['timestamp'],dtype={'model':'category','date':'string'})
    return df

def analyze(df,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    required={'timestamp','date','model','loss','VaR99','ES99','breach'}
    if not required.issubset(df):raise ValueError('Missing columns: '+str(required-set(df)))
    rows=[];years=[];daily=[]
    for model,g in df.groupby('model',observed=True,sort=True):
        g=g.sort_values('timestamp');b=np.asarray(g.breach,dtype=int)
        lruc,puc=kupiec(b);lrind,pind,trans=independence_test(b,g.timestamp)
        s=joint_score(g.loss,g.VaR99,g.ES99)
        rows.append({'model':str(model),'n':len(g),'breaches':int(b.sum()),'breach_rate':b.mean(),'UC_LR':lruc,'UC_p':puc,'IND_LR':lrind,'IND_p':pind,'CC_LR':lruc+lrind if np.isfinite(lrind) else np.nan,'CC_p':chi2.sf(lruc+lrind,2) if np.isfinite(lrind) else np.nan,'transitions':trans,'mean_joint_score':np.mean(s),'mean_pinball':g.pinball.mean() if 'pinball' in g else np.nan,'mean_es_identification':np.mean(g.es_identification) if 'es_identification' in g else np.nan})
        year=pd.DataFrame({'year':g.timestamp.dt.year.to_numpy(),'breach':b,'joint_score':s,'loss':g.loss.to_numpy(),'var':g.VaR99.to_numpy(),'es':g.ES99.to_numpy()})
        for yr,z in year.groupby('year'):
            years.append({'model':str(model),'year':int(yr),'n':len(z),'breaches':int(z.breach.sum()),'breach_rate':z.breach.mean(),'joint_score':z.joint_score.mean(),'mean_VaR':z['var'].mean(),'mean_ES':z.es.mean()})
        for date,z in g.assign(joint_score=s).groupby('date',sort=True):
            daily.append({'model':str(model),'date':str(date),'n':len(z),'breaches':int(z.breach.sum()),'joint_score':z.joint_score.mean(),'joint_score_sum':z.joint_score.sum(),'pinball_sum':z.pinball.sum() if 'pinball' in z else np.nan})
    summary=pd.DataFrame(rows).sort_values('mean_joint_score')
    summary.to_csv(out/'model_backtests.csv',index=False)
    pd.DataFrame(years).to_csv(out/'yearly_model_comparison.csv',index=False)
    pd.DataFrame(daily).to_csv(out/'daily_model_scores.csv',index=False)
    # Extreme Event Lab: unique events, ranked by loss, with contemporaneous forecasts.
    events=df.sort_values('loss',ascending=False).drop_duplicates('timestamp').head(40)[['timestamp','loss']]
    forecast=df[df.timestamp.isin(events.timestamp)].pivot(index='timestamp',columns='model',values=['VaR99','ES99'])
    flat=forecast.copy();flat.columns=['_'.join(x) for x in flat.columns]
    events=events.set_index('timestamp').join(flat).reset_index()
    events.to_csv(out/'extreme_events_top40.csv',index=False)
    # Daily loss severity and breaches, by model
    dmax=df.groupby('date',observed=True).loss.max().rename('maximum_five_minute_loss').reset_index()
    dmax.nlargest(40,'maximum_five_minute_loss').to_csv(out/'extreme_days_top40.csv',index=False)
    (out/'METHODOLOGY.json').write_text(json.dumps({'tests':'Kupiec unconditional coverage; Christoffersen first-order independence and conditional coverage on uninterrupted 5-minute transitions','pvalue_warning':'Asymptotic tests; dependence, regime variation, forecast estimation, multiple comparisons, and large sample sizes affect interpretation. Rejecting null does not prove economic significance.','joint_score':'((loss - VaR) * I(loss > VaR) / q + VaR) / ES + log(ES) - 1, with q=.01, positive loss VaR/ES. Compare means only for identical observations and units.','important':'No causality attribution to economic news; timestamps alone are insufficient. No model refitting in this module.'},indent=2))
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results-zip',default='STOCHASTIC_LAB_5YEAR_RESULTS.zip');ap.add_argument('--output-dir',default='v08_outputs')
    args=ap.parse_args();summary=analyze(load_saved(args.results_zip),args.output_dir)
    print(summary.to_string(index=False));print('SAVED',args.output_dir)
if __name__=='__main__':main()
