# Physics-Informed Machine Learning for Financial Market Dynamics and Risk Forecasting

MSc Physics semester project (five months). This project builds a framework that combines financial time-series analysis, physics-inspired features (entropy, Hurst exponent, diffusion scaling) and machine learning to forecast future 5-trading-day realized volatility of the S&P 500.

**Central research question:** do physics-inspired features provide a measurable out-of-sample improvement over conventional financial features in forecasting future realized volatility?

## Data
- S&P 500 Index (`^GSPC`), daily, 1 April 2011 to 1 April 2026 (3,772 trading days)
- Source: Yahoo Finance, via `python/download_sp500.py`. Raw data are **not committed**. Run the script to regenerate them.
- Full provenance, validation checks and decisions: [`docs/data_log.md`](docs/data_log.md)

## Definitions
- Daily log return: r_t = ln(P_t / P_{t-1})
- Feature: RV5_past_t = sqrt( sum_{i=0}^{4} r_{t-i}^2 )
- Target: RV_{t,t+5} = sqrt( sum_{i=1}^{5} r_{t+i}^2 ). Not annualized. All features at day t use information available up to and including day t only.

## Repository structure
```
data/raw/            raw downloads (git-ignored)
data/processed/      derived datasets (git-ignored)
mathematica/         notebooks (01_Data_Collection_and_EDA.nb)
python/              download script; ML code (later)
results/figures/     figures
results/tables/      result tables
results/models/      trained models (git-ignored)
docs/                data log, notes
```

## Reproducing Milestone 1
1. `pip install yfinance pandas openpyxl`
2. `python python/download_sp500.py`, then move `SP500_2011_2026.csv` into `data/raw/`
3. Open `mathematica/01_Data_Collection_and_EDA.nb` in Mathematica 13.0 and evaluate all cells

## Progress
- [x] Milestone 1: data collection, validation, log returns, stylized facts, realized volatility, target construction and aligned dataset
- [ ] Physics-inspired feature engineering
- [ ] Baseline and physics-augmented ML models
- [ ] Advanced models, explainability, regime analysis
