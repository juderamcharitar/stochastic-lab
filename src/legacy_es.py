"""Backfill ES diagnostics from V0.6's saved *real* historical GARCH VaR forecasts.
This is not the same as multi-year contiguous validation; it covers 11 sparse days.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm,t as tdist
from src.tailrisk_v07 import validate_es

def reconstruct_es(v06_output_dir):
    inp=Path(v06_output_dir)
    detail=pd.read_csv(inp/'walkforward_observations.csv')
    fits=pd.read_csv(inp/'walkforward_fits.csv').set_index('date')
    records=[]
    for model in ('GARCH Gaussian','GARCH Student-t'):
        group=detail[detail.model==model].copy()
        if group.empty:continue
        estimated=[]
        for date,day in group.groupby('date',sort=False):
            fit=fits.loc[date]
            mu=float(fit['mu']);df=float(fit['student_t_df'])
            if model=='GARCH Gaussian':
                critical=norm.ppf(.01);tailmean=-norm.pdf(critical)/.01
            else:
                x=tdist.ppf(.01,df);scale=np.sqrt((df-2)/df)
                critical=scale*x
                tailmean=-(df+x*x)/(df-1)*tdist.pdf(x,df)/.01*scale
            sigma=(day.VaR99.to_numpy()+mu)/(-critical)
            if (sigma<=0).any():raise ValueError('Inconsistent stored VaR forecasts')
            estimated.extend((-mu-sigma*tailmean).tolist())
        group['ES99']=estimated
        group['es_identification']=group['breach']*(group['loss']-group['ES99'])/.01
        records.append(group)
    full=pd.concat(records,ignore_index=True)
    return full,validate_es(full)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v06-output-dir',default='outputs');ap.add_argument('--output-dir',default='v07_outputs')
    a=ap.parse_args();df,summary=reconstruct_es(a.v06_output_dir)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'legacy_16day_es_observations.csv',index=False)
    summary.to_csv(out/'legacy_16day_es_summary.csv',index=False)
    print(summary.to_string(index=False))
if __name__=='__main__':main()
