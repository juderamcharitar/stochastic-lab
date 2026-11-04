"""Additive V1.0 held-out scenario research dashboard."""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

def render():
    st.divider();st.header('V1.0 · Monte Carlo Market Simulator')
    st.caption('Calibrated on 2021–2024 only. 2025 is an untouched holdout. Simulated 5-minute paths are hypothetical scenarios, not forecasts of future BTC prices.')
    root=Path('v10_outputs')
    f=root/'heldout_model_comparison.csv'
    if not f.exists():
        st.info('Run `py -m src.market_sim_v10 --returns-csv btc_5m_returns.csv --paths 10000` to generate these results.')
        return
    d=pd.read_csv(f)
    st.subheader('2025 historical reality check')
    st.dataframe(d[['model','observed_var_breaches','historical_days','observed_breach_rate','var99_logreturn_loss','es99_logreturn_loss','daily_return_crps','daily_return_pinball_1pct']],hide_index=True,use_container_width=True)
    st.plotly_chart(px.bar(d,x='model',y='daily_return_crps',title='Daily return CRPS — lower is better'),use_container_width=True)
    st.plotly_chart(px.bar(d,x='model',y='observed_breach_rate',title='Observed daily 99% VaR breach rate — target 1%'),use_container_width=True)
    st.subheader('Monte Carlo convergence of rare daily losses')
    c=root/'mc_convergence.csv'
    if c.exists():
        dd=pd.read_csv(c)
        st.plotly_chart(px.box(dd,x='paths',y='es99',points='all',title='Tail ES variation across independent simulation replications'),use_container_width=True)
        st.plotly_chart(px.box(dd,x='paths',y='var99',points='all',title='VaR variation across independent simulation replications'),use_container_width=True)
    grid=root/'risk_surface.csv'
    if grid.exists():
        import plotly.graph_objects as go
        surface=pd.read_csv(grid)
        x=sorted(surface.horizon_days.unique());y=sorted(surface.volatility_multiplier.unique())
        z=surface.pivot(index='volatility_multiplier',columns='horizon_days',values='es99').loc[y,x].values
        fig=go.Figure(go.Surface(x=x,y=y,z=z,colorscale='Plasma',lighting=dict(ambient=.4,diffuse=.8,roughness=.4)))
        fig.update_layout(title='Scenario sensitivity — 99% Expected Shortfall',scene=dict(xaxis_title='Horizon (days)',yaxis_title='Volatility multiplier',zaxis_title='Simulated ES (log return loss)',camera=dict(eye=dict(x=1.5,y=1.4,z=1.0))),height=550)
        st.plotly_chart(fig,use_container_width=True)
        st.caption('Simulated GJR scenario surface; not a historical parameter-robustness test. Same random shocks are reused across grid cells.')
    s=root/'scenario_samples.csv'
    if s.exists():
        ss=pd.read_csv(s)
        st.plotly_chart(px.histogram(ss,x='daily_logreturn',color='model',barmode='overlay',nbins=50,title='Sampled simulated daily log returns'),use_container_width=True)
    st.warning('Frozen 2024 calibration, synthetic daily scenarios and sparse 1% holdout tail events limit conclusions. No live trading validation or realized profits are implied.')
