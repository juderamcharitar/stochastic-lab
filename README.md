# STOCHASTIC LAB

**Bitcoin tail-risk research · GARCH / GJR-GARCH · Monte Carlo · 3D volatility terrain**

Research code and working manuscript analyzing Binance spot BTC/USDT five-minute returns from 2021–2025 with a later January–October 2026 chronological evaluation. The main comparison uses symmetric Student-t GARCH and asymmetric Student-t GJR-GARCH, with joint VaR/Expected Shortfall scores. The independent daily Monte Carlo experiment evaluates 2025 daily returns under a different protocol.

> **Working research, not peer-reviewed.** The paper reports the precise forecast protocols and known limitations. No profitability, causal inference, or live-trading claim is made.

## Key observations

| Protocol | Result | Qualification |
|---|---|---|
| 2022–2025 clean refit | GJR minus symmetric score: −0.004480 (60d), −0.004263 (120d), −0.003762 (240d) | 102/102 saved fits converged; development-period selection remains a limitation |
| Jan–Oct 2026 chronological holdout | −0.004356 (120d lookback); 87,552 five-minute observations over 304 UTC days | Favorable 7-day block-bootstrap interval; both models exceed the nominal 1% VaR breach rate |
| 2025 daily Monte Carlo | Symmetric Student-t GARCH lowest estimated CRPS | Frozen-parameter *daily* experiment, not directly comparable with five-minute joint VaR/ES |

## Explore

1. Download or clone the repo.
2. Open **`OPEN_THIS_NEON_TERRAIN.html`** for the standalone offline 3D view. It visualizes **16 selected historical days**, not the full five-year experiment. It intentionally includes a separate cinematic vertical-exaggeration mode.
3. For the Python dashboard:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

4. Run the test suite:

```bash
python -m unittest discover -s tests -v
```

## Data and reproducibility

Raw Binance monthly archives and the full five-year return CSV are **not distributed in this repository**. Official public monthly candles can be downloaded with:

```bash
python -m src.download_binance --start 2021-01 --end 2026-10
python -m src.multiyear --allow-gaps
```

To redo the clean development-period fit after producing `v07_outputs/btc_5m_returns.csv`:

```bash
python validation/full_refit_runner.py --returns-csv v07_outputs/btc_5m_returns.csv --windows 60,120,240 --start-year 2022 --output-dir clean_outputs
```

For the primary 2026 holdout, use a **separate output directory** and retain the frozen 120-day history / 90-day refit protocol:

```bash
python validation/full_refit_runner.py --returns-csv v07_outputs/btc_5m_returns.csv --windows 120 --refit-days 90 --start-year 2026 --output-dir holdout_2026_frozen
```

> The runner needs NumPy, pandas and SciPy; it assumes the repository root contains `src/robustness_v09.py`. Historical monthly files must be checked for integrity and missing intervals. The software refuses to silently fabricate returns across gaps. The commands above are intended to be rerun; the release does not claim they were rerun in a fresh external machine environment.

## Research artifacts

- [`paper/STOCHASTIC_LAB_V115_REVISED_MANUSCRIPT.pdf`](paper/STOCHASTIC_LAB_V115_REVISED_MANUSCRIPT.pdf) — ten-page working paper
- [`paper/HOLDOUT_AUDIT.md`](paper/HOLDOUT_AUDIT.md) — audit scope and qualifications
- [`validation/v113_clean_refit/`](validation/v113_clean_refit/) — clean development fit outputs
- [`validation/2026_holdout/`](validation/2026_holdout/) — 2026 evaluation outputs
- [`src/`](src/) — simulation and model code
- [`tests/`](tests/) — Python test suite
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — protocols, provenance, boundaries

## Important limitations

- Research is on Binance spot BTC/USDT, not other exchanges or assets.
- Some intervals are missing; the pipeline excludes returns spanning gaps.
- Published model ranks depend on metric, horizon, and calibration protocol.
- 2026 is a later chronological evaluation, **not** a fully blinded or preregistered trial.
- Conditional risk calibration remains imperfect and extreme losses can overwhelm VaR.
- The neon terrain has its own 16-day visualization subset, distinct from the risk-model evaluation.

## Credit and licensing

Research project prepared for **Jude Kriel Ramcharitar**. Confirm final paper authorship and any collaborators before publishing. **No open-source license is assigned yet**; select one explicitly before inviting reuse. Data belong to their respective publishers.
