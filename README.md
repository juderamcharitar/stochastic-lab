# STOCHASTIC LAB

**Bitcoin volatility research | GARCH and GJR-GARCH | Monte Carlo simulation | 3D volatility terrain**

STOCHASTIC LAB explores how well statistical volatility models estimate Bitcoin's downside risk. The project uses Binance BTC/USDT five-minute prices from 2021–2026 to compare symmetric Student-t GARCH against asymmetric Student-t GJR-GARCH. It also includes a separate daily Monte Carlo experiment and an interactive 3D visualization of historical volatility.

The main question is whether modeling asymmetric volatility improves forecasts of Value at Risk (VaR) and Expected Shortfall (ES). The results below come from historical evaluations, not live trading.

## Results

| Experiment | Finding | Context |
|---|---|---|
| 2022–2025 model comparison | GJR minus GARCH joint VaR/ES score: **−0.004480** (60-day training), **−0.004263** (120-day), **−0.003762** (240-day) | All 102 saved optimizer fits converged after the optimizer was corrected. These were development-period comparisons. |
| January–October 2026 evaluation | **−0.004356** using the preselected 120-day training window | 87,552 five-minute forecasts over 304 UTC days. The paired block-bootstrap interval favors GJR, although both models breach their nominal 1% VaR threshold too often. |
| 2025 daily Monte Carlo experiment | Symmetric Student-t GARCH had the lowest estimated CRPS | A separate daily-return experiment; its scores are not directly comparable with the five-minute VaR/ES results. |

Negative GJR-minus-GARCH joint scores favor the asymmetric model. The 2026 experiment extends the earlier evaluation chronologically, but it was not a fully blinded or preregistered study.

## Run the project

Clone the repository and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the dashboard:

```bash
python -m streamlit run app.py
```

For the standalone 3D view, open **`OPEN_THIS_NEON_TERRAIN.html`** in a browser. It uses 16 selected historical days, not all five years of model-testing data. A separate cinematic mode exaggerates vertical movement for visualization.

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Reproduce the data and model comparison

The full historical return CSV and original Binance monthly archives are not stored here. Download the monthly candles and reconstruct the gap-safe returns:

```bash
python -m src.download_binance --start 2021-01 --end 2026-10
python -m src.multiyear --allow-gaps
```

The pipeline checks historical coverage and excludes returns that would span missing intervals.

Run the corrected development-period comparison (the end-year cutoff prevents 2026 observations from entering the 2022–2025 evaluation):

```bash
python validation/full_refit_runner.py --returns-csv v07_outputs/btc_5m_returns.csv --windows 60,120,240 --start-year 2022 --end-year 2025 --output-dir clean_outputs
```

Run the separate 2026 comparison using the previously selected 120-observed-day training window and 90-observed-day refit schedule:

```bash
python validation/full_refit_runner.py --returns-csv v07_outputs/btc_5m_returns.csv --windows 120 --refit-days 90 --start-year 2026 --output-dir holdout_2026_frozen
```

The refit runner requires NumPy, pandas, and SciPy and uses `src/robustness_v09.py` from the repository. The saved results are included, but the complete pipeline has not yet been independently reproduced on another machine.

## Files

- [Research manuscript (PDF)](paper/STOCHASTIC_LAB_V115_REVISED_MANUSCRIPT.pdf)
- [2026 holdout audit](paper/HOLDOUT_AUDIT.md)
- [Corrected 2022–2025 results](validation/v113_clean_refit/)
- [2026 evaluation results](validation/2026_holdout/)
- [Model and simulation code](src/)
- [Tests](tests/)
- [Reproducibility notes](REPRODUCIBILITY.md)

## Scope and limitations

The analysis covers Binance spot BTC/USDT rather than multiple exchanges or assets. Some historical candles are missing, and the processing excludes returns spanning those gaps. Model rankings also depend on the forecast horizon and scoring method. In particular, a better average joint VaR/ES score does not imply perfect tail-risk calibration: both models exceeded the 1% VaR breach target in the 2026 evaluation.

This is a working research project, not a peer-reviewed publication. It does not demonstrate a profitable trading strategy.

## Attribution and licensing

**Sole creator and author: Jude Kriel Ramcharitar.**

Carnegie Mellon University · Independent research.

No open-source license has been selected. Original market data remain subject to their publishers' terms.
