"""Resumable, clean per-window refit of the STOCHASTIC LAB V0.9 protocol.
Run from this script's directory, with STOCHASTIC_V09_ROOT pointing to the extracted V0.9 project.
Output is a distinct run, never silently overwrites existing results.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('STOCHASTIC_V09_ROOT',str(HERE.parent))).resolve()
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from src import robustness_v09 as original
from robust_optimizer import repair

def hashfile(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--returns-csv',default=None)
 ap.add_argument('--output-dir',default=str(HERE/'clean_outputs'))
 ap.add_argument('--windows',default='60,120,240')
 ap.add_argument('--refit-days',type=int,default=90)
 ap.add_argument('--start-year',type=int,default=2022)
 ap.add_argument('--end-year',type=int,default=None,help='Inclusive end of evaluation period; preserve earlier training history')
 ap.add_argument('--block-days',default='3,7,14,30,60')
 ap.add_argument('--draws',type=int,default=1500)
 ap.add_argument('--force',action='store_true',help='Explicitly replace completed outputs for a window')
 args=ap.parse_args()
 path=Path(args.returns_csv or ROOT/'btc_5m_returns.csv').resolve()
 if not path.exists():ap.error(f'Missing returns CSV: {path}')
 out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
 windows=[int(x) for x in args.windows.split(',')]
 if any(w<30 for w in windows):ap.error('Each training window must be >=30 observed days')
 returns=original.load_returns(path)
 if args.end_year is not None:
  if args.end_year < args.start_year:ap.error('--end-year must be >= --start-year')
  returns=returns.loc[returns.index.year <= args.end_year]
  if returns.empty:ap.error('No historical returns at or before --end-year')
 original.fit_model=repair
 manifest={'source_path':str(path),'source_sha256':hashfile(path),'refit_days':args.refit_days,'start_year':args.start_year,'end_year':args.end_year,'windows':windows,'fit_engine':'robust_optimizer.repair SLSQP constrained multistart','code_sha256':{'runner':hashfile(Path(__file__)),'repair':hashfile(HERE/'robust_optimizer.py'),'original':hashfile(ROOT/'src'/'robustness_v09.py')},'status':{}}
 daily_list=[];fit_list=[]
 for w in windows:
  directory=out/f'window_{w}';directory.mkdir(exist_ok=True)
  daily_file=directory/'daily_scores.csv';fit_file=directory/'model_refits.csv'
  if daily_file.exists() and fit_file.exists() and not args.force:
   print(f'REUSE completed window {w} (use --force to recompute)',flush=True)
  else:
   print(f'RUNNING clean window={w}, training history={w} observed days; start-year={args.start_year}',flush=True)
   t=time.monotonic()
   daily,fits=original.run(returns,train_days=(w,),refit_days=args.refit_days,start_year=args.start_year,fit_obs=100000,progress=True)
   if not fits.converged.all():raise RuntimeError(f'Nonconverged optimizer result in window {w}')
   for fr,d in ((daily,'daily_scores.csv'),(fits,'model_refits.csv')):
    tmp=directory/(d+'.tmp');fr.to_csv(tmp,index=False);os.replace(tmp,directory/d)
   print(f'SAVED window={w}; {len(fits)} converged refits; elapsed={time.monotonic()-t:.1f}s',flush=True)
  daily=pd.read_csv(daily_file);fit=pd.read_csv(fit_file)
  if daily.empty or fit.empty:raise RuntimeError(f'Empty output for window {w}')
  if not fit.converged.astype(bool).all():raise RuntimeError(f'Nonconverged saved fit window {w}')
  daily_list.append(daily);fit_list.append(fit)
  manifest['status'][str(w)]={'fits':len(fit),'daily_rows':len(daily),'converged':int(fit.converged.sum()),'daily_sha256':hashfile(daily_file),'fit_sha256':hashfile(fit_file)}
  (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
 combined=pd.concat(daily_list,ignore_index=True);fits=pd.concat(fit_list,ignore_index=True)
 combined.to_csv(out/'all_daily_scores.csv',index=False)
 fits.to_csv(out/'all_model_refits.csv',index=False)
 original.summarize(combined).to_csv(out/'yearly_comparison.csv',index=False)
 bs=[]
 for b in [int(v) for v in args.block_days.split(',')]:
  frame=original.paired_bootstrap(combined,draws=args.draws,seed=20261104+b,block_days=b)
  bs.append(frame)
 bootstrap=pd.concat(bs,ignore_index=True)
 bootstrap.to_csv(out/'bootstrap_sensitivity.csv',index=False)
 print('\nCONVERGENCE\n',fits.groupby(['train_days','model']).converged.agg(['sum','count']).to_string())
 print('\nBOOTSTRAP\n',bootstrap.to_string(index=False))
 print(f'\nCOMPLETE: {out}')

if __name__=='__main__':main()
