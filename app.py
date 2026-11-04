"""STOCHASTIC LAB interactive Plotly dashboard; reads validated precomputed surfaces."""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.robustness import create_figure, create_heatmap, summarize_surface

st.set_page_config(page_title='STOCHASTIC LAB',page_icon='🌊',layout='wide')
st.markdown('''<style>
.stApp{background:#080d19;color:#e6ecf5}
[data-testid="stMetric"]{background:#131f32;padding:14px;border-radius:12px;border:1px solid #263952}
</style>''',unsafe_allow_html=True)
st.title('STOCHASTIC LAB')
st.caption('MONTE CARLO × RISK RESEARCH × ROLLING ROBUSTNESS')
st.info('Animated surfaces use scored historical evaluation dates, not fluid physics. Lower negative log likelihood is better. Never select and validate a model on the same dates.')
choices=list(Path('outputs').glob('rolling_surface.csv'))
file=st.file_uploader('Load another rolling_surface.csv',type=['csv'])
if file is not None: df=pd.read_csv(file)
elif choices: df=pd.read_csv(choices[0])
else:
    st.warning('No surface yet. Run: py src\\robustness.py --demo or provide historical data with --data-dir data')
    st.stop()
required={'date','alpha','beta','loss','n_test','n_train'}
if not required.issubset(df.columns):st.error('Invalid surface data schema');st.stop()
st.caption('SOURCE: '+(Path('outputs/SURFACE_PROVENANCE.txt').read_text().splitlines()[0] if file is None and Path('outputs/SURFACE_PROVENANCE.txt').exists() else 'UPLOADED CSV - provenance not independently verified'))
summary=summarize_surface(df)
a,b,c=st.columns(3)
a.metric('Evaluation dates',str(df.date.nunique()))
b.metric('Parameter grid',str(len(df)//df.date.nunique()))
c.metric('Median plateau fraction',f'{summary.plateau_fraction_heuristic.median():.1%}')
tab1,tab2,tab3,tab4,tab5=st.tabs(['🏔 Real Volatility Mountains','⛰ GARCH Robustness Terrain','🌊 Plotly GARCH Surface','🗺 2D Robustness Map','📊 Evaluation Diagnostics'])
with tab1:
    import streamlit.components.v1 as components
    volpath=Path('outputs/volatility_mountains.html')
    if volpath.exists():
        components.html(volpath.read_text(encoding='utf-8'),height=890,scrolling=True)
        st.caption('Observed Binance-based volatility spikes; two axes are UTC half-hour and rolling lookback. Cinematic heights are exaggerated and percentile-clipped. This is NOT the GARCH robustness surface.')
    else:
        st.info('Run py src\\export_volatility_terrain.py with your Binance trade archives in data/ to generate the real mountains.')
with tab2:
    import streamlit.components.v1 as components
    from src.export_terrain import export
    terrain_path=Path('outputs/neon_terrain.html')
    if file is not None:
        uploaded_csv=Path('outputs/uploaded_rolling_surface.csv')
        uploaded_csv.write_bytes(file.getvalue())
        export(uploaded_csv,terrain_path)
    components.html(terrain_path.read_text(encoding='utf-8'),height=890,scrolling=True)
    st.caption('Cinematic relief is exaggerated and monotonic. Research mode preserves comparable global z-scale. Intermediate frames are interpolated.')
with tab3:
    st.plotly_chart(create_figure(df),use_container_width=True)
    st.caption('Use Play/Pause, the date slider, or drag the camera to rotate. Color and z-axis scales are held fixed across frames.')
with tab4:
    d=st.select_slider('Evaluation date',options=sorted(df.date.unique()),value=sorted(df.date.unique())[0])
    sub=df[df.date==d].pivot(index='alpha',columns='beta',values='loss')
    st.plotly_chart(go.Figure(go.Contour(x=sub.columns,y=sub.index,z=sub.values,colorscale='Turbo',colorbar=dict(title='Loss'))).update_layout(template='plotly_dark',height=540,xaxis_title='β',yaxis_title='α'),use_container_width=True)
with tab5:
    st.dataframe(summary,use_container_width=True,hide_index=True)
    st.line_chart(summary.set_index('date')[['best_loss']])
    st.download_button('Download model sensitivity results',data=df.to_csv(index=False),file_name='rolling_surface.csv')
st.caption('Scientific caveat: the minimum loss in each evaluation window is selected using its outcomes. It is descriptive, not a fully independent prospective forecast. Partial/nonconsecutive days need broader validation.')

st.divider()
st.header('V0.6 · Walk-Forward Tail-Risk Lab')
st.caption('Chronological day-by-day evaluation. GARCH filter resets each observed day; no gap-spanning state. Lower pinball score is better; 99% VaR targets ~1% exceedances.')
wf_path=Path('outputs/walkforward_summary.csv')
if wf_path.exists():
    results=pd.read_csv(wf_path)
    st.dataframe(results.style.format({'breach_rate':'{:.2%}','target':'{:.2%}','mean_pinball':'{:.8f}','avg_VaR99':'{:.4%}'}),use_container_width=True,hide_index=True)
    col_a,col_b=st.columns(2)
    with col_a:
        st.image('outputs/walkforward_coverage.png',caption='Observed exceedance rates across held-out dates')
    with col_b:
        st.image('outputs/walkforward_scores.png',caption='Strictly out-of-sample quantile loss')
    st.download_button('Download walk-forward summary',data=results.to_csv(index=False),file_name='walkforward_summary.csv')
    st.info('Sparse/nonconsecutive days and small numbers of breaches make model rankings preliminary. Student-t residual fit is two-stage, not joint MLE.')
else:
    st.info('Generate the walk-forward study: py -m src.walkforward --data-dir data --output-dir outputs')

st.divider()
st.header('V0.7 · Multi-Year Tail Risk & Monte Carlo Validation')
st.caption('The V0.7 multi-year results are produced only after downloading and verifying contiguous Binance monthly archives. Existing 16-day results are clearly marked preliminary.')
from pathlib import Path as _Path
v07=_Path('v07_outputs')
mc=v07/'mc_option_convergence.csv'
if mc.exists():
    st.subheader('Numerical convergence: synthetic Black–Scholes benchmark')
    tab1,tab2=st.tabs(['Pricing convergence','Gaussian tail convergence'])
    with tab1:
        res=pd.read_csv(mc)
        st.line_chart(res.pivot(index='n_paths',columns='method',values='rmse'))
        st.dataframe(res,hide_index=True,use_container_width=True)
        st.caption('RMSE across independent replications against an analytical Black–Scholes call price; not forecasts on Binance data.')
    with tab2:
        tfile=v07/'mc_gaussian_tail_convergence.csv'
        if tfile.exists():
            tdf=pd.read_csv(tfile)
            st.line_chart(tdf.pivot(index='n_paths',columns='method',values='ES_rmse'))
            st.dataframe(tdf,hide_index=True,use_container_width=True)
            st.caption('Known Gaussian VaR/Expected Shortfall reference; antithetic sampling is not guaranteed to reduce tail estimator variance.')
legacy=v07/'legacy_16day_es_summary.csv'
if legacy.exists():
    st.subheader('Real Binance, sparse-day Expected Shortfall diagnostics')
    st.dataframe(pd.read_csv(legacy),use_container_width=True,hide_index=True)
    st.warning('Only 11 evaluation dates. The day-block bootstrap interval is exploratory; this is NOT a formal conditional ES backtest or multi-year evidence.')
full=v07/'var_es_validation.csv'
if full.exists():
    st.subheader('Contiguous multi-year backtest')
    st.dataframe(pd.read_csv(full),use_container_width=True,hide_index=True)
    coverage=v07/'coverage.json'
    if coverage.exists():
        import json as _json
        c=_json.loads(coverage.read_text())
        st.caption(f"Verified months: {c.get('months')} | Bars: {c.get('candles'):,} | Fully contiguous: {c.get('continuous')}")
else:
    st.info('Multi-year evaluation not run yet. First download official monthly BTCUSDT 5-minute archives using py -m src.download_binance --start 2021-01 --end 2025-12, then run py -m src.tailrisk_v07.')

from src.dashboard_v08 import render as render_v08
render_v08()

from src.dashboard_v09 import render as render_v09
render_v09()

from src.dashboard_v10 import render as render_v10
render_v10()
