"""Additive V0.9 research panels. Preserves V0.8 UI and neon viewer unchanged."""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

def render():
    root=Path('v09_outputs')
    st.divider();st.header('V0.9 · Rolling Model Robustness')
    st.caption('Real BTC/USDT 5-minute returns, Binance 2021–2025. 2021 supplies prior training history; evaluations use 2022–2025. Student-t symmetric GARCH and asymmetric GJR-GARCH; refit every 90 observed days. This is risk forecasting—not a trading strategy.')
    yearly=root/'yearly_comparison.csv';ci=root/'paired_day_bootstrap.csv';fits=root/'model_refits.csv'
    if not yearly.exists():
        st.info('Run `py -m src.robustness_v09 --returns-csv STOCHASTIC_LAB_5YEAR_RESULTS.zip` to generate V0.9 results. Existing precomputed results should also appear here.')
        return
    years=pd.read_csv(yearly);summary=pd.read_csv(ci)
    st.subheader('Walk-forward joint VaR / ES scores')
    st.caption('Lower joint score is better. Each training-window comparison uses the same evaluation observations.')
    st.plotly_chart(px.line(years,x='year',y='mean_joint_score',color='model',line_dash='train_days',markers=True,labels={'mean_joint_score':'Average joint score'}),use_container_width=True)
    st.subheader('Paired moving-block uncertainty')
    st.dataframe(summary,hide_index=True,use_container_width=True)
    st.caption('Asymmetric minus symmetric score: negative favors GJR. Confidence intervals resample seven-day blocks, an exploratory approximation; refit windows and market regimes create longer dependencies.')
    st.subheader('VaR calibration by evaluation year')
    years['breach_pct']=100*years.breach_rate
    st.plotly_chart(px.line(years,x='year',y='breach_pct',color='model',line_dash='train_days',markers=True),use_container_width=True)
    st.caption('99% VaR target is 1%; both under- and over-exceedance matter. Variance forecast state resets at each day and observed data gap.')
    st.dataframe(years,hide_index=True,use_container_width=True)
    if fits.exists():
        p=pd.read_csv(fits)
        st.subheader('Fitted parameter stability')
        for param in ('alpha','beta','gamma','df'):
            st.plotly_chart(px.line(p,x='date',y=param,color='model',line_dash='train_days',markers=True,title=param),use_container_width=True)
    st.warning('Two-stage Student-t innovation fitting, calendar-day resets, finite optimization budgets, overlapping training windows, and approximate bootstrap inference limit causal or general claims. No prospective trading performance is established.')
