"""Rolling, exploratory GARCH parameter sensitivity surfaces.

Days are distinct independent evaluation sequences. No volatility recursion crosses
missing archive boundaries or even observed calendar-day boundaries. The same grid
is scored on later dates; choosing a winner from these surfaces incurs selection bias.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp


def make_returns_from_prices(prices, horizon=5):
    from research import contiguous_returns
    return contiguous_returns(prices, horizon)


def build_surface(returns: pd.Series, train_days: int = 4,
                  alpha_grid=None, beta_grid=None):
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.index.tz is None:
        raise ValueError('Returns must have a timezone-aware datetime index')
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError('Returns require sorted unique timestamps')
    alpha = np.asarray(alpha_grid if alpha_grid is not None else np.linspace(.01,.22,23),dtype=float)
    beta = np.asarray(beta_grid if beta_grid is not None else np.linspace(.65,.985,25),dtype=float)
    if np.any(alpha<=0) or np.any(beta<0) or np.any(alpha>=1) or np.any(beta>=1):
        raise ValueError('Invalid parameter grid')
    A,B=np.meshgrid(alpha,beta,indexing='ij')
    valid=(A+B<.995)
    days=sorted(set(returns.index.date))
    if len(days) < train_days+2: raise ValueError('Need at least train_days+2 distinct dates')
    frames=[]
    for k in range(train_days,len(days)):
        train=returns[np.isin(returns.index.date,days[k-train_days:k])].to_numpy(dtype=float)
        test=returns[returns.index.date==days[k]].to_numpy(dtype=float)
        if len(train)<100 or len(test)<20: continue
        mu=float(train.mean());s=float(train.std(ddof=1))
        if s<=1e-12: continue
        # Genuine forward calendar split; fixed mean and scale fitted from prior dates.
        z=(test-mu)/s
        omega=1-A-B
        v=np.ones_like(A)
        loss=np.zeros_like(A)
        count=0
        for r in z:
            # Relative Gaussian NLL in normalized coordinates; constant terms cancel
            safe_v=np.maximum(v,1e-10)
            loss += .5*(np.log(safe_v)+r*r/safe_v)
            v=omega+A*r*r+B*v
            count+=1
        loss/=count
        loss=np.where(valid,loss,np.nan)
        frames.extend({'date':str(days[k]),'alpha':float(alpha[i]),'beta':float(beta[j]),
                       'loss':float(loss[i,j]) if valid[i,j] else np.nan,
                       'n_test':count,'n_train':len(train),'train_start':str(days[k-train_days]),
                       'train_end':str(days[k-1])}
                      for i in range(len(alpha)) for j in range(len(beta)))
    if not frames: raise ValueError('Insufficient returns in evaluation dates')
    return pd.DataFrame(frames)


def summarize_surface(df):
    rows=[]
    for date,g in df.groupby('date',sort=True):
        a=g.dropna(subset=['loss']);best=a.loc[a.loss.idxmin()]
        tolerance=.02 # 2% of median absolute loss; not a formal statistical interval
        med=float(a.loss.median()); threshold=float(best.loss)+tolerance*max(abs(med),.01)
        broad=float((a.loss<=threshold).mean())
        rows.append({'date':date,'best_alpha':best.alpha,'best_beta':best.beta,
                     'best_loss':best.loss,'plateau_fraction_heuristic':broad,
                     'n_test':int(best.n_test),'n_train':int(best.n_train)})
    return pd.DataFrame(rows)


def create_figure(df, dark=True):
    import plotly.graph_objects as go
    dates=sorted(df.date.unique());a=sorted(df.alpha.unique());b=sorted(df.beta.unique())
    surfaces=[]
    for d in dates:
        sub=df[df.date==d].pivot(index='alpha',columns='beta',values='loss').reindex(index=a,columns=b)
        surfaces.append(sub.to_numpy(dtype=float))
    allz=np.concatenate([q[np.isfinite(q)] for q in surfaces]);
    zmin=float(np.quantile(allz,.01));zmax=float(np.quantile(allz,.99))
    # Fixed camera and z/color scales prevent false-looking movement due to rescaling.
    def surface(z): return go.Surface(x=b,y=a,z=z,colorscale='Turbo',cmin=zmin,cmax=zmax,
                             colorbar=dict(title='NLL'),connectgaps=False)
    fig=go.Figure(data=[surface(surfaces[0])],frames=[go.Frame(data=[surface(z)],name=d) for d,z in zip(dates,surfaces)])
    steps=[dict(method='animate',args=[[d],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))],label=d) for d in dates]
    fig.update_layout(template='plotly_dark' if dark else 'plotly_white',title='STOCHASTIC LAB | Rolling GARCH sensitivity',
                      scene=dict(xaxis_title='β · persistence',yaxis_title='α · shock response',zaxis_title='Mean Gaussian NLL',
                                 zaxis=dict(range=[zmin,zmax]),aspectratio=dict(x=1.25,y=1,z=.65),
                                 camera=dict(eye=dict(x=1.55,y=1.5,z=.85))),
                      margin=dict(l=10,r=10,b=10,t=58),height=670,
                      updatemenus=[dict(type='buttons',x=0.02,y=1.08,showactive=False,buttons=[
                        dict(label='▶ Play',method='animate',args=[None,dict(frame=dict(duration=850,redraw=True),transition=dict(duration=250),fromcurrent=True,mode='immediate')]),
                        dict(label='❚❚ Pause',method='animate',args=[[None],dict(frame=dict(duration=0,redraw=False),mode='immediate')])])],
                      sliders=[dict(active=0,steps=steps,x=.08,y=-.04,len=.87,currentvalue=dict(prefix='Evaluation date: '))])
    return fig


def create_heatmap(df):
    import plotly.graph_objects as go
    dates=sorted(df.date.unique());a=sorted(df.alpha.unique());b=sorted(df.beta.unique())
    vals=[df[df.date==d].pivot(index='alpha',columns='beta',values='loss').reindex(index=a,columns=b).values for d in dates]
    fig=go.Figure(data=[go.Heatmap(x=b,y=a,z=vals[0],colorscale='Turbo',colorbar=dict(title='NLL'),hovertemplate='β=%{x:.3f}<br>α=%{y:.3f}<br>NLL=%{z:.4f}<extra></extra>')],
                  frames=[go.Frame(data=[go.Heatmap(x=b,y=a,z=v,colorscale='Turbo')],name=d) for d,v in zip(dates,vals)])
    fig.update_layout(template='plotly_dark',title='2D Parameter Robustness Map',height=550,
                      xaxis_title='β · persistence',yaxis_title='α · shock response')
    return fig


def main():
    p=argparse.ArgumentParser(description='Real-data rolling GARCH sensitivity surfaces')
    p.add_argument('--data-dir',default='data');p.add_argument('--output-dir',default='outputs');p.add_argument('--demo',action='store_true')
    p.add_argument('--max-files',type=int);p.add_argument('--train-days',type=int,default=4)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if a.demo:
        from research import demo_prices
        prices=demo_prices(days=12)
        provenance='SYNTHETIC DEMONSTRATION - NOT REAL TRADING DATA'
    else:
        from research import load_prices
        prices,_=load_prices(Path(a.data_dir),a.max_files)
        provenance='BINANCE HISTORICAL DATA'
    ret=make_returns_from_prices(prices)
    surf=build_surface(ret,train_days=a.train_days)
    surf.to_csv(out/'rolling_surface.csv',index=False)
    summ=summarize_surface(surf);summ.to_csv(out/'surface_diagnostics.csv',index=False)
    create_figure(surf).write_html(out/'animated_3d_surface.html',include_plotlyjs=True,full_html=True)
    create_heatmap(surf).write_html(out/'animated_heatmap.html',include_plotlyjs=True,full_html=True)
    (out/'SURFACE_PROVENANCE.txt').write_text(provenance+'\nExploratory OOS sensitivity, selection-biased minima; reset variance to 1 each evaluation day.\n')
    print('SOURCE:',provenance,'\nDates:',surf.date.nunique(),'\nGrid:',len(surf)//surf.date.nunique(),'\nSaved:',out.resolve())
    print(summ.to_string(index=False))
if __name__=='__main__':main()
