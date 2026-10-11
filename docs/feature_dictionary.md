# Feature Dictionary — features_v1

Created in `mathematica/02_Feature_Engineering.nb` on 11 October 2026.
File: `data/processed/features_v1.csv` (git-ignored; regenerate by running notebook 02).

## Notation
- P_t: S&P 500 Close on trading day t
- r_t = ln(P_t / P_{t-1}): daily log return
- All features at row t use only prices up to and including day t. The target uses only returns t+1 … t+5.
- Time is measured in trading sessions. Nothing is annualized.

## Columns

| Column | Definition | Look-back (returns) | Units | Role |
|---|---|---|---|---|
| Date | Trading day t (YYYY-MM-DD) | n/a | n/a | index |
| Close | P_t | n/a | index points | **reference only, not a model input** (D15) |
| r | r_t | 1 | log return | feature |
| r_lag1 | r_{t-1} | 2 | log return | feature |
| r_lag2 | r_{t-2} | 3 | log return | feature |
| abs_r | \|r_t\| | 1 | per day | feature (HAR daily) |
| RV5_past | sqrt( sum_{i=0}^{4} r_{t-i}^2 ) | 5 | 5-day total | feature (HAR weekly); naive baseline forecast |
| RV22_past | sqrt( sum_{i=0}^{21} r_{t-i}^2 ) | 22 | 22-day total | feature (HAR monthly) |
| sigma20 | sample SD of r_{t-19} … r_t, divisor 19 (demeaned) | 20 | per day | feature |
| M10 | ln(P_t / P_{t-10}) = sum_{i=0}^{9} r_{t-i} | 10 | log return | feature (momentum) |
| RV5_target | sqrt( sum_{i=1}^{5} r_{t+i}^2 ) | uses future t+1 … t+5 | 5-day total | **target** |

Model inputs (8): r, r_lag1, r_lag2, abs_r, RV5_past, RV22_past, sigma20, M10.

## Rows
- Return-days t = 22 … 3766: the first day with a full 22-return look-back to the last day with 5 future returns
- 3,745 rows; first Wed 4 May 2011, last Wed 25 Mar 2026; 0 missing values

## Decisions
- **D15:** Close is excluded from model inputs. The price level is non-stationary, and a model given it could learn the time period rather than volatility dynamics, then fail on unseen price levels.

## Verification (all passed)
| Check | Result |
|---|---|
| Rebuild of r_t from raw CSV matches notebook 01 | 3,771 returns; r_1 = 0.000345179, mean 0.000423318, SD 0.010927 |
| Hand checks at first valid day | RV5_past 0.00482945, RV22_past 0.0259194, sigma20 0.00557322: all equal direct recomputation |
| Quadratic variation, RV22 | sqrt(mean(RV22^2)/22) = 0.010939 vs daily SD 0.010927 |
| Lag alignment | r_lag1 / r_lag2 on Wed 24 Aug 2011 equal the returns of Tue 23 Aug and Mon 22 Aug 2011 |
| Momentum two-formula check (all days) | max \|sum of returns − log price ratio\| = 1.8 × 10^-15 |
| Target identity in table | RV5_target[row k] == RV5_past[row k+5]: True |
| Cross-notebook check vs `sp500_rv5_dataset.csv` | max abs difference 0 on all 3,745 shared rows |
| **Truncation leakage test** (28 cutoffs: 3 fixed + 25 random, SeedRandom[1]) | features recomputed from prices up to day T equal full-sample values to machine precision: max \|Δ\| = 1.4 × 10^-17 |
| Negative control: planted look-ahead feature (centred mean of \|r\| over t−2 … t+2) | detected at 28 of 28 cutoffs; max \|Δ\| = 0.0056 |

## Descriptive statistics (full sample — descriptive only, NOT used for feature selection)

Extremes: max sigma20 = 0.0617 (Fri 27 Mar 2020); min M10 = −0.266 (Wed 18 Mar 2020).

Correlation with RV5_target:

| Feature | Pearson | Spearman |
|---|---|---|
| r | −0.173 | −0.147 |
| r_lag1 | −0.180 | −0.135 |
| r_lag2 | −0.154 | −0.123 |
| abs_r | 0.496 | 0.322 |
| RV5_past | 0.666 | 0.552 |
| RV22_past | 0.593 | 0.532 |
| sigma20 | 0.597 | 0.542 |
| M10 | −0.458 | −0.357 |

Notable feature–feature correlations: RV22_past–sigma20 0.99 (near-duplicate), RV5_past–RV22_past 0.79, RV5_past–M10 −0.46. Corr(r, r_lag1) = −0.12 and corr(r, r_lag2) = 0.08 reproduce the lag-1 and lag-2 return autocorrelations found in notebook 01.

Interpretation (hypotheses, to be tested out-of-sample):
- Recent volatility is the strongest single predictor; the 5-day window beats the 22-day window for a 5-day-ahead target.
- Negative correlations for returns and M10 are consistent with the leverage effect (falls are followed by higher volatility).
- Spearman < Pearson for every volatility feature: part of the linear association is driven by extreme high-volatility episodes.
- RV22_past and sigma20 are near-duplicates: harmless for tree models; makes linear coefficients unstable (Ridge addresses this). Both are kept, as specified in D6 and D8.

Tables: `results/tables/feature_target_correlations.csv`, `results/tables/feature_correlation_matrix.csv`
Figures: `results/figures/07_volatility_features.png`, `results/figures/08_momentum.png`
