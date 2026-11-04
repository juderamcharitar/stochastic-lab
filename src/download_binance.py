"""Windows-friendly official Binance monthly ZIP downloader with optional SHA-256 verification."""
from __future__ import annotations
import argparse, calendar, hashlib, os, time, urllib.error, urllib.request
from pathlib import Path
BASE='https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/'

def months(first,last):
    y,m=map(int,first.split('-'));ye,me=map(int,last.split('-'))
    if (y,m)>(ye,me):raise ValueError('Start after end')
    while (y,m)<=(ye,me):
        yield f'{y:04d}-{m:02d}'
        m+=1
        if m==13:m=1;y+=1

def fetch(url,dest,retries=3):
    temp=dest.with_suffix(dest.suffix+'.download')
    for attempt in range(retries):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'StochasticLab/0.7 research'})
            with urllib.request.urlopen(req,timeout=50) as r,open(temp,'wb') as f:
                while True:
                    block=r.read(1024*1024)
                    if not block:break
                    f.write(block)
            temp.replace(dest);return
        except (TimeoutError,OSError,urllib.error.URLError) as e:
            if attempt==retries-1:raise
            time.sleep(2*(attempt+1))

def download_range(first,last,target,skip_existing=True):
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    from src.multiyear import check_sha256
    for month in months(first,last):
        name=f'BTCUSDT-5m-{month}.zip';p=target/name;cp=target/(name+'.CHECKSUM')
        if not p.exists() or not skip_existing:fetch(BASE+name,p)
        if not cp.exists() or not skip_existing:fetch(BASE+name+'.CHECKSUM',cp)
        try: digest=check_sha256(p,cp)
        except ValueError:
            p.unlink(missing_ok=True)
            fetch(BASE+name,p)
            digest=check_sha256(p,cp)
        print(f'VERIFIED {month}  {p.stat().st_size/1e6:.2f} MB  SHA-256 {digest[:12]}... ',flush=True)

def main():
    p=argparse.ArgumentParser(description='Official Binance BTCUSDT 5m monthly candles (not raw trades)')
    p.add_argument('--start',default='2021-01');p.add_argument('--end',default='2025-12');p.add_argument('--data-dir',default='data/monthly_5m')
    a=p.parse_args();download_range(a.start,a.end,a.data_dir)
if __name__=='__main__':main()
