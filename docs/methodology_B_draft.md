# Methodology (draft)

*Draft by Person B, 12 Oct 2026. Items marked **[CONFIRM]** need checking against the notebooks or a decision from A.*

## 1. Research question

Can physics-inspired features improve machine-learning forecasts of next-week stock-market volatility? We answer it in two phases:

1. **Baseline:** how well can volatility be forecast *without* physics?
2. **Physics:** do physics-inspired features measurably improve on that baseline, out of sample, using identical splits, models and metrics?

## 2. Data

- **Source:** S&P 500 index (^GSPC), daily prices from Yahoo Finance via `python/download_sp500.py`. Mathematica's `FinancialData` would not return history, hence the Python download.
- **Period:** 1 Apr 2011 to 1 Apr 2026, 3,772 trading days.
- **Quality checks:** no duplicates, no missing prices, all highs and lows consistent with open and close. All date gaps are explained by weekends and holidays, including the 2012 Hurricane Sandy closure. One zero-volume day (24 May 2023) was set to missing; its prices were kept.
- **Independent verification:** the 31 Mar 2026 close (6528.52) matched a Yahoo screenshot, and Person B rebuilt the dataset independently on a second machine with identical results (3,772 rows, same close).
- Full provenance is in `docs/data_log.md`. Raw data is kept out of Git (`.gitignore`).

## 3. Returns and stylized facts

- Daily log returns: $r_t = \ln P_t - \ln P_{t-1}$, verified to sum exactly to the total 15-year log change.
- **Fat tails:** 28 days beyond 4 standard deviations, against about 0.24 expected under a normal distribution. The largest move (16 Mar 2020) was nearly 12 standard deviations.
- **Volatility clustering:** the sign of returns is nearly unpredictable, but their size is strongly autocorrelated for weeks. This is what makes volatility forecastable.

## 4. Target

Five-day realized volatility over the **next** five trading days:

$$RV5^{\text{future}}_t = \sqrt{\sum_{i=1}^{5} r_{t+i}^2}$$

It was proved to equal the backward-looking $RV5^{\text{past}}$ shifted five days earlier (zero difference). Because it contains future information, it is **never** used as an input.

## 5. Features (`features_v1.csv`, 3,745 rows)

| Group | Feature | Definition |
|---|---|---|
| HAR (daily) | $\lvert r_t\rvert$ | absolute return today |
| HAR (weekly) | RV5_past | $\sqrt{\sum_{i=0}^{4} r_{t-i}^2}$ |
| HAR (monthly) | RV22_past | 22-day realized volatility **[CONFIRM scaling used in Mathematica]** |
| Volatility | sigma20 | 20-day standard deviation of returns |
| Returns | $r_t, r_{t-1}, r_{t-2}$ | today's and two lagged returns |
| Trend | momentum | 10-day momentum **[CONFIRM exact definition]** |

**Leakage control (truncation test):** all data after a cutoff was deleted and every feature recomputed. No feature changed (largest difference about $10^{-17}$, rounding noise). A deliberately "cheating" feature was planted as a negative control and was caught at all 28 cutoffs.

## 6. Train, validation and test splits

Chronological, no shuffling:

| Set | Share | Period (approx.) | Includes |
|---|---|---|---|
| Train | 65% | May 2011 to Dec 2020 | COVID crash |
| Validation | 15% | Jan 2021 to Mar 2023 | 2022 bear market |
| Test | 20% | 2023 to Mar 2026 | April 2025 shock |

**Purging:** the last 5 rows before each boundary are removed, because their 5-day targets reach into the next period. All tuning uses validation only. **The test set is opened exactly once, at the end.**

## 7. Models (Phase 1 baseline)

In order of complexity:

1. **Training mean:** a constant forecast.
2. **Naive:** next week equals this week ($RV5^{\text{past}}$).
3. **HAR** (Corsi 2009): linear regression on daily, weekly and monthly volatility. Also Ridge regression on all features.
4. **Random Forest.**
5. **XGBoost.**

## 8. Evaluation (`python/metrics.py`)

- **MAE, RMSE, R²** on volatility.
- **QLIKE** (Patton 2011), on variance: $L = \frac{\sigma^2}{h} - \ln\frac{\sigma^2}{h} - 1$, where $\sigma^2$ is realized and $h$ is forecast variance. It is robust to noise in the volatility proxy and penalises under-prediction more than over-prediction.
- **Diebold–Mariano test** (1995): is one model's loss significantly lower than another's? Newey–West variance with 4 lags (overlapping 5-day targets) and the Harvey–Leybourne–Newbold small-sample correction.
- **Calm vs turbulent periods** scored separately. **[CONFIRM threshold, e.g. above or below the training median of RV5]**

`metrics.py` passes 6 self-tests on simulated data, including a 5.0% false-alarm rate for the DM test on equally good forecasts.

## 9. Preliminary result (validation only)

From `python/quick_baseline_har.py`, which rebuilds the HAR features in Python (3,745 rows, matching the Mathematica table):

| Model | RMSE | R² | QLIKE |
|---|---|---|---|
| Training mean | 0.01304 | −0.21 | 0.813 |
| Naive | 0.01177 | 0.02 | 0.896 |
| HAR | 0.01033 | 0.24 | 0.470 |

HAR beats naive significantly (DM: QLIKE p = 0.004, MSE p = 0.0006). Preliminary: one validation period, Python features not yet compared value by value with `features_v1.csv`.

## 10. Phase 2: physics features (planned)

- **Hurst exponent** by detrended fluctuation analysis (`python/hurst_dfa.py`): measures persistence. Validated on simulated data only so far (7 self-tests, e.g. H = 0.51 for random-walk steps, 0.69 and 0.31 for known H = 0.7 and 0.3).
- **Entropy** (disorder) and **diffusion-scaling** measures.
- Physics features will be computed on rolling past windows only and pass the same truncation test before use.
- Same splits, models and metrics as Phase 1. The single question: do they improve out-of-sample forecasts, judged by the DM test?

## 11. Reproducibility

- Python virtual environment with pinned versions in `requirements.txt`. Machines: Python 3.13.3 (A) and 3.14.6 (B).
- All code in Git; each script reviewed by the other team member before use.

## References

- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
- Peng, C.-K., et al. (1994). Mosaic organization of DNA nucleotides. *Physical Review E*, 49(2), 1685–1689.
