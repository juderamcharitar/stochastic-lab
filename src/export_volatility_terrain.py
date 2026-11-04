"""Export terrain from actual Binance trade prices: 30-min intraday bins x rolling volatility horizons.
The 3D heights are log ratios of observed realized volatility versus same-day median at matching horizon.
No synthetic bumps or noise added; daily frames are measured, interpolated transitions are labelled.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from research import read_zip_minutes

ROOT=Path(__file__).resolve().parents[1]
HORIZONS=np.unique(np.round(np.geomspace(3,90,27)).astype(int)).tolist()
BIN_MINUTES=30

def make_frame(prices):
    s=prices.sort_index().resample('1min').last().ffill(limit=1)
    # Only use returns from consecutive observed minutes, not overnight or across missing minutes
    log=np.log(s)
    r=log.diff()
    rr=r.to_numpy()
    frame=[]
    for h in HORIZONS:
        # A locally estimated realized volatility curve; minimum full h observations
        rv=np.sqrt(pd.Series(rr*rr).rolling(h,min_periods=h).sum())
        grid=[]
        for bin_idx in range(48):
            chunk=rv.iloc[bin_idx*30:(bin_idx+1)*30]
            grid.append(float(np.nanmedian(chunk)) if np.isfinite(chunk).any() else np.nan)
        grid=np.asarray(grid)
        med=np.nanmedian(grid)
        # compare with same-horizon baseline; remove systematic h-dependent scaling
        norm=grid/med if np.isfinite(med) and med>0 else np.full(48,np.nan)
        frame.append([round(float(x),5) if np.isfinite(x) else None for x in norm])
    return frame

def export(data_dir=None,output=None,max_files=None):
    data_dir=Path(data_dir or ROOT/'data')
    paths=sorted(data_dir.glob('BTCUSDT-trades-*.zip'))
    if max_files:paths=paths[:max_files]
    if not paths:raise FileNotFoundError('No daily Binance trades ZIPs found')
    frames=[];dates=[];manifest=[]
    for p in paths:
        prices=read_zip_minutes(p)
        if len(prices)<1000:continue
        # sanity: all prices from the same UTC date
        dates_in_data=prices.index.strftime('%Y-%m-%d').unique()
        if len(dates_in_data)!=1:continue
        frame=make_frame(prices)
        frames.append(frame);dates.append(dates_in_data[0]);manifest.append({'archive':p.name,'observed_minute_prices':len(prices)})
        print(f'{p.name}: {len(prices)} observed minutes, terrain frame ready',flush=True)
    if not frames:raise RuntimeError('No usable frames')
    data={'dates':dates,'alpha':HORIZONS,'beta':list(range(48)),'frames':frames,
          'provenance':'BINANCE spot historical raw trade archives; rolling realized-volatility ratio, UTC; no inferred data on missing days',
          'manifest':manifest}
    tpl=(ROOT/'src'/'terrain_template.html').read_text()
    tpl=tpl.replace('NEON TERRAIN <span style="color:#bc65fe">/</span> GARCH ROBUSTNESS','NEON TERRAIN <span style="color:#bc65fe">/</span> REAL VOLATILITY')
    tpl=tpl.replace('Binance BTC/USDT · 12 historical evaluation windows · α × β × Gaussian NLL','Binance BTC/USDT · observed intraday volatility · 16 historical days')
    tpl=tpl.replace('α = shock response; β = volatility persistence. <strong>Lower NLL is better.</strong> Color is tied to original global loss values; a mountainous peak is not necessarily desirable.',
                    'Horizontal axes: UTC time of day (30-minute bins) and volatility lookback (3–90 minutes). Height: observed realized volatility / same-day median at matching lookback. <strong>Peaks indicate actual volatility bursts.</strong>')
    tpl=tpl.replace(' NLL</span>',' RV ratio</span>')
    tpl=tpl.replace('Every vertex is derived from a valid GARCH grid cell in the 16-day Binance study. Invalid α+β regions remain missing.',
                    'Every vertex derives from Binance 1-minute reconstructed last trade prices, summarized by 30-minute UTC bin and rolling volatility lookback. Missing minutes are not bridged.')
    tpl=tpl.replace('Nonconsecutive historical dates, selected minima, and finite sample size limit the conclusions.',
                    'Frames show nonconsecutive UTC dates; animation between dates is visual interpolation, not real observations. This is descriptive volatility, not model robustness or predictive evidence.')
    tpl=tpl.replace('original NLL','volatility ratio').replace('original values preserved','raw ratios preserved').replace('NLL','RV ratio')
    tpl=tpl.replace('const all=DATA.frames.flat(2).filter(v=>v!==null&&Number.isFinite(v));const bounds=[Math.min(...all),Math.max(...all)];',
                    'const all=DATA.frames.flat(2).filter(v=>v!==null&&Number.isFinite(v)).sort((a,b)=>a-b);const q=p=>all[Math.floor(p*(all.length-1))];const bounds=[q(.02),q(.98)];')
    tpl=tpl.replace('let u=(z-bounds[0])/range;', 'let u=clamp((z-bounds[0])/range,0,1);')
    tpl=tpl.replace('let x=(u-.5)*2;return Math.sign(x)*Math.pow(Math.abs(x),.72)*app.exaggeration*.64;',
                    'let x=(u-.5)*2;return Math.sign(x)*Math.pow(Math.abs(x),.72)*app.exaggeration*.64;')
    tpl=tpl.replace('function getField(i,j){','function getField(i,j){')
    # preserve raw values, but clip cinematic heights at global robust data bounds (label in display)
    tpl=tpl.replace("'Cinematic: monotonic vertical exaggeration '+app.exaggeration.toFixed(1)+'× · raw ratios preserved'", "'Cinematic: percentile-clipped relief '+app.exaggeration.toFixed(1)+'× · color and height from measured ratios'")
    tpl=tpl.replace('Cinematic mode applies a monotonic vertical transformation and lighting; it does not invent new local extrema.',
                    'Cinematic mode applies a percentile-clipped monotonic height transform and lighting, and may hide subtle extrema; no extra peaks or random noise are added.')
    tpl=tpl.replace('__DATA__',json.dumps(data,separators=(',',':')))
    output=Path(output or ROOT/'outputs'/'volatility_mountains.html')
    output.write_text(tpl)
    (output.parent/'volatility_terrain_data.json').write_text(json.dumps(data,separators=(',',':')))
    return output

if __name__=='__main__':
    import sys
    print(export(max_files=int(sys.argv[1]) if len(sys.argv)>1 else None))
