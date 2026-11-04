"""Verified Binance monthly 5m archives -> contiguous UTC five-minute log returns.
No HTTP needed for analysis; download.py handles optional online acquisition separately.
"""
from __future__ import annotations
import argparse, calendar, hashlib, io, json, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore']
NAME=re.compile(r'^BTCUSDT-5m-(\d{4})-(\d{2})\.zip$')

def time_unit(values):
    a=np.asarray(values,dtype='int64')
    m=int(np.median(a))
    if 10**14<m<10**17:return 'us'
    if 10**11<m<10**14:return 'ms'
    raise ValueError(f'Unsupported Binance timestamp magnitude: {m}')

def check_sha256(path, checksum_path):
    expected=checksum_path.read_text(encoding='utf8').strip().split()[0].lower()
    if not re.fullmatch('[0-9a-f]{64}',expected): raise ValueError('Invalid checksum sidecar')
    with path.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=expected:raise ValueError(f'Checksum mismatch: {path.name}')
    return digest

def load_month(path, require_checksum=True):
    match=NAME.match(path.name)
    if not match:raise ValueError('Expected BTCUSDT-5m-YYYY-MM.zip')
    yr,mo=map(int,match.groups())
    if mo<1 or mo>12:raise ValueError('Invalid month')
    sidecar=Path(str(path)+'.CHECKSUM')
    if require_checksum:
        if not sidecar.exists():raise FileNotFoundError(f'Missing {sidecar.name}; download it or use --allow-unverified explicitly')
        check_sha256(path,sidecar)
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.csv')]
        if len(names)!=1:raise ValueError(f'Expected exactly one CSV in {path.name}')
        with z.open(names[0]) as f:
            df=pd.read_csv(f,header=None,names=COLS,usecols=[0,4],dtype=str)
    # Binance files are generally headerless; support a header row when present.
    df=df[pd.to_numeric(df.open_time,errors='coerce').notna()]
    if df.empty:raise ValueError(f'No kline data in {path.name}')
    stamps=pd.to_numeric(df.open_time,errors='raise').astype('int64')
    unit=time_unit(stamps)
    times=pd.to_datetime(stamps,unit=unit,utc=True)
    closes=pd.to_numeric(df.close,errors='raise').astype(float)
    if not np.isfinite(closes).all() or (closes<=0).any():raise ValueError(f'Invalid closes in {path.name}')
    idx=pd.DatetimeIndex(times)
    month_start=pd.Timestamp(yr,mo,1,tz='UTC')
    month_end=month_start+pd.offsets.MonthBegin(1)
    if (idx<month_start).any() or (idx>=month_end).any():raise ValueError(f'Unexpected timestamps in {path.name}')
    if idx.has_duplicates:raise ValueError(f'Duplicate candles in {path.name}')
    if not idx.is_monotonic_increasing:raise ValueError(f'Unsorted candles in {path.name}')
    if (idx.asi8 % (5*60*10**9)).any():raise ValueError('Not aligned to 5m grid')
    expected_count=int((month_end-month_start)/pd.Timedelta(minutes=5))
    s=pd.Series(closes.to_numpy(),index=idx,name='close')
    return s,{'file':path.name,'year_month':f'{yr:04d}-{mo:02d}','actual_bars':len(s),'expected_bars':expected_count,'missing_bars':expected_count-len(s),'verified_checksum':sidecar.exists() and require_checksum,'timestamp_unit':unit,'first':str(idx.min()),'last':str(idx.max())}

def load_history(folder,require_checksum=True,strict=True):
    files=sorted(Path(folder).glob('BTCUSDT-5m-????-??.zip'))
    if not files:raise FileNotFoundError(f'No monthly 5m archives in {folder}')
    observations=[];stats=[]
    for p in files:
        s,meta=load_month(p,require_checksum)
        observations.append(s);stats.append(meta)
    periods=pd.PeriodIndex([d['year_month'] for d in stats],freq='M')
    full=pd.period_range(periods.min(),periods.max(),freq='M')
    missing_months=[str(p) for p in full.difference(periods)]
    combined=pd.concat(observations).sort_index()
    if combined.index.has_duplicates:raise ValueError('Overlapping duplicate timestamps across archives')
    delta=combined.index.to_series().diff().dropna()
    bad=delta[delta!=pd.Timedelta(minutes=5)]
    coverage={'months':len(stats),'range_months':len(full),'missing_months':missing_months,'candles':len(combined),'missing_candles':int(sum(max(0,d['missing_bars']) for d in stats)), 'gap_count':int(len(bad)),'max_gap_minutes':float(bad.max()/pd.Timedelta(minutes=1)) if len(bad) else 0,'files':stats,'continuous':not missing_months and not len(bad) and not sum(max(0,d['missing_bars']) for d in stats)}
    if strict and not coverage['continuous']:
        raise ValueError('Not fully continuous: '+json.dumps({k:coverage[k] for k in ['missing_months','missing_candles','gap_count','max_gap_minutes']}))
    # Never compute returns across gaps in permissive mode either.
    logs=np.log(combined)
    returns=logs.diff().where(combined.index.to_series().diff()==pd.Timedelta(minutes=5)).dropna()
    returns.name='log_return_5m'
    return returns,coverage

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='data/monthly_5m');ap.add_argument('--output-dir',default='v07_outputs');ap.add_argument('--allow-unverified',action='store_true');ap.add_argument('--allow-gaps',action='store_true')
    a=ap.parse_args()
    r,manifest=load_history(a.data_dir,not a.allow_unverified,not a.allow_gaps)
    out=Path(a.output_dir);out.mkdir(exist_ok=True,parents=True)
    r.to_frame().to_csv(out/'btc_5m_returns.csv')
    (out/'coverage.json').write_text(json.dumps(manifest,indent=2))
    print(f"{len(r):,} returns; {manifest['months']} months; continuous={manifest['continuous']}")
if __name__=='__main__':main()
