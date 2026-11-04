"""Additive V0.8 panels to preserve previously working Stochastic Lab app."""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px


def render():
    root=Path('v08_outputs')
    st.divider();st.header('V0.8 · Scientific Validation Labs')
    st.caption('Audited saved forecasts: 2021–2025. New asymmetric experiment is one frozen 2024→2025 split. Monte Carlo convergence is synthetic.')
    tabs=st.tabs(['🚨 Extreme Event Lab','🧪 Conditional VaR & ES','📉 Asymmetric GARCH','🎲 Monte Carlo Tail Risk'])
    with tabs[0]:
        p=root/'extreme_events_top40.csv';q=root/'event_volatility_context.csv'
        if p.exists():
            data=pd.read_csv(p);st.dataframe(data.head(20),hide_index=True,use_container_width=True)
            if q.exists():
                c=pd.read_csv(q)
                st.scatter_chart(c,x='pre_60m_rv',y='post_60m_rv',size='loss',color=None)
                st.dataframe(c[['timestamp','loss','pre_60m_rv','post_60m_rv','complete_pre','complete_post']].head(15),hide_index=True)
            st.warning('Large losses are observed events, not necessarily identifiable news events. No causal attribution from timestamp alone.')
        else:st.info('Run: py -m src.validation_v08 --results-zip STOCHASTIC_LAB_5YEAR_RESULTS.zip')
    with tabs[1]:
        p=root/'model_backtests.csv';y=root/'yearly_model_comparison.csv'
        if p.exists():
            d=pd.read_csv(p);st.dataframe(d,hide_index=True,use_container_width=True)
            st.bar_chart(d.set_index('model')[['mean_joint_score']]);st.caption('Lower joint VaR/ES score is better, provided models share the same sample and loss convention.')
            if y.exists():
                ys=pd.read_csv(y);st.plotly_chart(px.line(ys,x='year',y='breach_rate',color='model',markers=True),use_container_width=True)
            st.warning('Asymptotic coverage p-values assume approximations that can be sensitive to serial dependence, many observations, and model refitting. Check effect sizes.')
        else:st.info('Validation results not generated yet.')
    with tabs[2]:
        f=root/'experimental_asymmetric_2025.csv'
        if f.exists():
            d=pd.read_csv(f);st.dataframe(d,hide_index=True,use_container_width=True)
            st.caption('Parameters trained using the latest 35,000 returns in 2024, held fixed in 2025; this is not walk-forward GJR validation.')
        else:st.info('Run: py -m src.asymmetric_v08 --results-zip STOCHASTIC_LAB_5YEAR_RESULTS.zip')
    with tabs[3]:
        f=root/'mc_tail_convergence.csv'
        if f.exists():
            d=pd.read_csv(f);st.plotly_chart(px.line(d,x='paths',y='rmse',color='method',markers=True,log_x=True),use_container_width=True)
            st.dataframe(d,hide_index=True,use_container_width=True)
            st.caption('Independent synthetic Gaussian benchmark with known 99% ES; importance sampling uses exact normal likelihood weights and known VaR threshold. Not crypto forecast performance.')
        else:st.info('Run: py -m src.monte_carlo_v08')
