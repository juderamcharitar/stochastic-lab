# Reproducibility and provenance

## Input provenance
- Source: Binance public `spot/monthly/klines/BTCUSDT/5m/` archives. The original analysis downloaded 60 months (2021–2025); subsequently 10 more (Jan–Oct 2026). The data downloader checks SHA-256 files.
- Full five-year returns: 525,667 valid five-minute observations and 7 reported discontinuities. Combined through Oct 2026: 613,219 valid returns. Returns crossing missing candles are excluded.
- Four extreme five-minute intervals were rechecked against Binance one-minute archives; their independent publisher checksums were not available during that cross-resolution check.
- Avoid mixing experimental horizons: five-minute joint VaR/ES comparisons and a separately calibrated one-day MC distribution test.

## Model comparisons
- Development: 2022–2025, 60/120/240 *observed trading-day* windows, 90-day refit cadence, 102 fitted models in total, all saved with convergence flag true.
- Primary chronological evaluation: Jan–Oct 2026, 120-day training, 90-day refits, 304 days and 87,552 common observations per specification.
- Corrected model fit uses a constrained multistart optimization procedure; check `validation/robust_optimizer.py`. Student-t degrees of freedom are fitted after Gaussian QMLE volatility fitting, **not** by fully joint Student-t MLE.
- Selected score differences and block-bootstrap intervals are descriptive evidence from this protocol; not proof of universal superiority.
- Frozen 2026 evaluation was chosen after development but cannot be described as fully blinded because some 2026 markets were explored elsewhere.

## What this release verifies
- This package includes the saved development and 2026 holdout outputs plus the latest manuscript.
- Existing 62 unit tests passed in the packaging environment. A separate full re-run of all historical fits was completed on the user's laptop and the archived outputs have been consistency-checked in prior work; the release package itself is not a fresh third-party replication.
- This version was prepared with source integrity manifest SHA-256 hashes. These hashes document this package, not a signed external attestation.

## Build environment
- Dependency minimums are in `requirements.txt`; **these are not exact resolved versions**.
- CI runs unit tests on Python 3.11; numerical optimizer results may vary across SciPy versions and systems.
- To create a true environment lock, run `python -m pip freeze > environment-local.txt` in the verified execution environment and separately archive the Python version/platform. Do not pretend the package minimums are lockfile pins.

## Other research limitations
- The original one-minute archives were not cross-verified against exchange-independent trade data.
- Simulated jump mixtures were exploratory; the synthetic importance-sampling benchmark is not an empirical BTC forecast test.
- Refitting, filtering and score details must be reviewed in source before external replication.
