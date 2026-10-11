# Data Log — Physics-Informed ML for Financial Market Dynamics

## Environment
- Wolfram Mathematica 13.0.1 for Microsoft Windows (64-bit), released 28 January 2022
- Python download script: `download_sp500.py` using `yfinance` 1.7.0

## Dataset D1: S&P 500 daily prices

| Item | Value |
|---|---|
| Ticker | `^GSPC` (S&P 500 Index) |
| Source | Yahoo Finance, via Python `yfinance` 1.7.0 |
| Retrieval date | 11 October 2026 |
| Requested range | start `2011-04-01`, end `2026-04-02` (end date is exclusive, so 1 April 2026 is included) |
| Frequency | Daily trading sessions |
| Raw file | `data/raw/SP500_2011_2026.csv` (do not edit) |
| Columns | Date, Open, High, Low, Close, Adj Close, Volume |
| Observations | 3,772 trading days |
| First observation | Fri 1 April 2011 (Close 1332.41) |
| Last observation | Wed 1 April 2026 (Close 6575.32) |

### Processing applied by the download script
- Flattened the two-level column headers returned by `yfinance`
- Converted timestamps to plain dates (YYYY-MM-DD)
- Rounded all values to 2 decimal places

### Why `FinancialData` was not used
`FinancialData["^GSPC", "Name"]` worked, but historical price requests failed with "Internet download of data for FinancialData failed". This happened even for a short test range (2 January 2025 to 10 January 2025), while `URLRead["https://www.google.com"]` returned status 200. We therefore switched to the CSV workflow.

## Validation results (Mathematica, 11 October 2026)

| Check | Result |
|---|---|
| Rows × columns (including header) | 3,773 × 7 |
| Dates strictly increasing | True |
| Duplicate dates | 0 |
| Non-numeric values | 0 |
| Prices ≤ 0 | 0 |
| OHLC violations (Low ≤ Open, Close ≤ High) | 0 |
| Close ≠ Adj Close | 0, so the two columns are identical. Close is used as the price variable. |
| Zero-volume days | 1 (see Note 1) |
| Calendar gaps {days: count} | 1: 2948, 2: 40, 3: 682, 4: 100, 5: 1 (total 3,771 = 3,772 − 1) |
| Gaps ≥ 5 days | One: Fri 26 Oct 2012 → Wed 31 Oct 2012 (Hurricane Sandy market closure, 29–30 Oct 2012) |
| Cross-check against Yahoo website | 31 Mar 2026 Close 6528.52 matches the Yahoo Finance page |

### Note 1: Zero-volume day
- Date: 24 May 2023 (data row 3057)
- Its prices (Open 4132.96, High 4132.96, Low 4103.98, Close 4115.24) are consistent with the neighbouring days, so the price data are kept.
- An index cannot have zero traded volume on an open trading day, so this is treated as a **missing volume value** in the provider's data, not a true zero.
- Decision: keep the row (removing it would distort returns). If Volume is ever used as a feature, treat this value as missing.

### Note 2: Hurricane Sandy closure
- The market was closed on 29–30 October 2012. The return from 26 Oct to 31 Oct 2012 therefore spans two closed sessions, but it is still one trading-session step.
- Decision: no adjustment. This is a genuine market closure, not missing data.

## Derived series R1: daily log returns (Mathematica, 11 October 2026)
- Definition: r_t = ln(P_t / P_{t-1}), using Close. Each return is dated at day t, using only the prices on t and t−1.
- Time convention: trading sessions, not calendar days. Weekend and holiday returns count as one step.
- Count: 3,771 returns, first dated Mon 4 Apr 2011
- Checks passed: first return 0.000345179 matches a hand calculation; sum of returns 1.59633 = ln(P_last / P_first)
- Mean 0.000423318, standard deviation 0.010927 (daily)
- Largest fall: Mon 16 Mar 2020, −0.127652. Largest rise: Wed 9 Apr 2025, +0.0908949.
- Figures: results/figures/01_close_linear.png, 01_close_log.png, 02_log_returns.png

## EDA E1: stylized facts of daily log returns (Mathematica, 11 October 2026)
| Quantity | Result | Gaussian reference |
|---|---|---|
| Skewness | −0.633527 | 0 |
| Kurtosis | 17.3807 | 3 |
| Days with \|z\| > 4 | 28 | 0.239 expected |
| ACF of \|r\|, lags 1–20 | ≈ 0.34–0.39 at lags 1–2, decaying slowly to ≈ 0.17 at lag 20; all far outside the ±0.032 band | ≈ 0 |
| ACF of r, lags 1–20 | Small but several outside the ±0.032 band: lag 1 ≈ −0.12, lag 2 ≈ +0.08, alternating ≈ ±0.08–0.09 at lags 6–9 | ≈ 0 |

- Interpretation: returns are fat-tailed and negatively skewed, and volatility is strongly clustered and persistent.
- Open item (not yet tested): the significant raw-return autocorrelations are suspected to come mostly from a few extreme days (especially Mar 2020). Also, the ±1.96/√n band assumes independent returns, so it is too narrow when volatility clusters. This should be checked with a robust test or by excluding 2020 before any conclusion is drawn.
- Figures: results/figures/03_return_distribution.png, 04_acf_returns_vs_abs.png

## Derived series V1: backward-looking 5-day realized volatility (Mathematica, 11 October 2026)
- Definition: RVpast_t = sqrt( sum_{i=0}^{4} r_{t-i}^2 ). Not demeaned, not annualized. Dated at day t, the last day of its window, so it is known at the close of day t.
- Count: 3,767 values (3,771 − 4). First value dated Fri 8 Apr 2011, using returns dated 4–8 Apr 2011 (alignment verified).
- Formula check: first value 0.00482945 matches a direct recalculation.
- Quadratic-variation check: sqrt(mean(RV^2)/5) = 0.0109322 vs daily SD 0.010927
- Min 0.00136219, median 0.0162352, max 0.200982 (Wed 18 Mar 2020)
- Figure: results/figures/05_rv5_backward.png

## Target T1: future 5-day realized volatility (Mathematica, 11 October 2026)
- Definition: RV_{t,t+5} = sqrt( sum_{i=1}^{5} r_{t+i}^2 ). Not demeaned, not annualized. Dated at day t, the day the forecast is made. It uses returns from t+1 to t+5 only, never r_t.
- Count: 3,766 values (3,771 − 5). The last 5 return-days have no target, because their future is outside the sample.
- First target: Mon 4 Apr 2011, built from returns dated 5, 6, 7, 8 and 11 Apr 2011 (verified)
- Last target: Wed 25 Mar 2026, built from returns dated 26, 27, 30 and 31 Mar, and 1 Apr 2026 (verified)
- Shift identity: target_t = RVpast_{t+5}. Checked exactly: rvFwd == Rest[rv5] is True, with max abs difference 0.
- Figure: results/figures/06_target_vs_backward_rv.png (Feb–Apr 2020, showing the target leading by 5 trading days)

## Dataset P1: aligned modelling table (Mathematica, 11 October 2026)
- File: `data/processed/sp500_rv5_dataset.csv` (3,762 rows + header, 5 columns)
- Columns: Date (YYYY-MM-DD), Close (P_t), LogReturn (r_t), RV5_past (returns t−4…t, a feature), RV5_target (returns t+1…t+5, the target)
- Rows kept: return-days t = 5 … 3766, the days with both a complete backward window and a complete future window
- First row: Fri 8 Apr 2011 (Close 1328.17, RV5_past 0.00482945, RV5_target 0.00917429)
- Last row: Wed 25 Mar 2026 (Close 6591.9, RV5_past 0.0203277, RV5_target 0.0385215). The Close matches the Yahoo Finance page.
- Non-numeric values: 0
- Leakage/alignment test: RV5_target[row k] == RV5_past[row k+5] for all k → True
- Round-trip export check: re-imported dimensions {3763, 5}; max abs difference 0
- Descriptive, in-sample correlation (not a forecast result): corr(RV5_past, RV5_target) = 0.666033

## Milestone 1 status: data collection, returns, realized volatility and target construction COMPLETE (11 October 2026)

## Cleaning operations
- None applied to the raw file. All checks were done in Mathematica on the imported data.
