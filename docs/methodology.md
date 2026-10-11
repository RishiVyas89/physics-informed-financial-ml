# Methodology and Baseline Results

Physics-Informed ML for Financial Market Dynamics and Risk Forecasting.
Phase 1: baseline ML pipeline (completed 12 October 2026).

See also `data_log.md` (data provenance and validation), `feature_dictionary.md` (feature definitions and checks) and `roadmap.md` (decisions D1–D17).

---

## 1. Research question
Can physics-inspired features (e.g. Hurst exponent, entropy, diffusion scaling) measurably improve out-of-sample forecasts of future S&P 500 realized volatility, compared with conventional financial features?

Phase 1 establishes the conventional baseline that Phase 2 must beat.

## 2. Data
- S&P 500 index (`^GSPC`), daily Close, 1 Apr 2011 – 1 Apr 2026, 3,772 trading days (Yahoo Finance via `yfinance`; retrieved 11 Oct 2026)
- Validated for duplicates, gaps, non-numeric and non-positive values, OHLC consistency; all calendar gaps explained by weekends, holidays and the Hurricane Sandy closure (Oct 2012); one zero-volume day treated as missing volume (prices unaffected)
- Time measured in trading sessions; nothing annualized

## 3. Target
Future 5-trading-day realized volatility, dated at the forecast day t:

  RV_{t,t+5} = sqrt( sum_{i=1}^{5} r_{t+i}^2 ),   r_t = ln(P_t / P_{t-1})

Verified identity: RV_{t,t+5} equals the backward 5-day realized volatility at day t+5 (max |difference| = 0).

## 4. Features (`features_v1`, 8 inputs)
r_t, r_{t-1}, r_{t-2}, |r_t|, RV5_past, RV22_past, sigma20 (20-day sample SD), M10 = ln(P_t/P_{t-10}). Full definitions in `feature_dictionary.md`.

The HAR subset (|r_t|, RV5_past, RV22_past) follows Corsi (2009). The price level is excluded (non-stationary).

## 5. Leakage controls
1. All features at row t use prices up to day t only; the target uses returns t+1 … t+5 only.
2. **Truncation test:** all features recomputed from prices truncated at day T equal the full-sample values at T, for 28 cutoffs (max |Δ| = 1.4 × 10⁻¹⁷, machine precision). A planted look-ahead feature was detected at 28/28 cutoffs (negative control).
3. **Independent re-implementation:** every feature and the target were recomputed in Python (pandas) from raw prices and matched Mathematica to ≤ 1.8 × 10⁻¹⁵.
4. **Chronological split with purging** (Section 6): no shuffling; rows whose 5-day target window crosses a split boundary are removed.
5. Scaling (Ridge) is fitted inside a pipeline on the fitting data only.
6. Hyperparameters are chosen on validation; the test set is evaluated once, after all choices were fixed.

## 6. Data split

| Split | Rows | Dates | Target mean | Target max |
|---|---|---|---|---|
| Train | 2,429 | 2011-05-04 → 2020-12-24 | 0.0189 | 0.201 |
| Purged | 10 | 5 rows from 2020-12-28; 5 rows from 2023-03-22 | – | – |
| Validation | 556 | 2021-01-05 → 2023-03-21 | 0.0244 | 0.061 |
| Test | 750 | 2023-03-29 → 2026-03-25 | 0.0176 | 0.122 |

The three periods differ in volatility regime: validation (incl. 2022 bear market) is the most volatile on average; train contains the most extreme episode (COVID-19, 2020); test is the calmest on average but includes the April 2025 shock.

## 7. Models and training protocol
| Model | Inputs | Tuning on validation (RMSE) |
|---|---|---|
| Naive | RV5_past ("next week = this week") | – |
| Train mean | constant | – |
| HAR (OLS) | \|r_t\|, RV5_past, RV22_past | – |
| Linear regression | all 8 | – |
| Ridge | all 8, standardized | α ∈ {10⁻³ … 10⁴}; chosen α = 1000 |
| Random Forest | all 8 | depth, min leaf, max features (24 configs); chosen depth 3, min leaf 5, max features 1.0, 500 trees |
| XGBoost | all 8 | depth, learning rate, min child weight (12 configs) + early stopping; chosen depth 4, lr 0.1, mcw 10, 17 trees |

Protocol: fit on train → select on validation → refit the chosen configuration on train + validation → predict test once. Seed 42. Software: Python 3.13, scikit-learn 1.9.1, XGBoost 3.4.1, statsmodels 0.15.0 (exact versions in `requirements.txt`); Mathematica 13.0.1.

All tuned models selected strongly regularized settings, consistent with the regime difference between train and validation. Grid-edge choices (RF depth 3, XGBoost depth 4) were not pursued further: the top validation models differed by < 0.5 % (decision D16).

## 8. Evaluation
- MAE, RMSE, R² (vs. test mean), R²_oos (vs. the train+val mean, a forecast known in advance)
- QLIKE on variances: y²/f² − ln(y²/f²) − 1 (Patton, 2011), robust to noise in the volatility proxy
- Diebold–Mariano tests vs. naive for squared error and QLIKE: loss differential regressed on a constant with HAC (Newey–West, 5 lags) standard errors, because overlapping 5-day targets make the losses autocorrelated
- Regime breakdown by terciles of RV5_past (known at forecast time; cut-offs 0.01207 and 0.02139 from train+val)

## 9. Baseline results

### 9.1 Validation (models fitted on train only)
RMSE relative to naive: Ridge 0.817, XGBoost 0.818, Random Forest 0.821, Linear 0.836, HAR 0.878, naive 1, train mean 1.110.

HAR coefficients on train (HAC t-statistics): constant 0.0046; daily 0.186 (z = 2.17); weekly 0.528 (z = 5.45); monthly 0.071 (z = 2.02). The weekly term dominates, as is typical in the HAR literature.

### 9.2 Test (750 forecasts, 29 Mar 2023 – 25 Mar 2026; evaluated once)

| Model | MAE | RMSE | R² | R²_oos | QLIKE | RMSE / naive | QLIKE / naive |
|---|---|---|---|---|---|---|---|
| Linear (all 8) | 0.005589 | 0.008968 | 0.356 | 0.382 | 0.3959 | 0.771 | 0.447 |
| Ridge | 0.005618 | 0.009027 | 0.347 | 0.374 | 0.3780 | 0.776 | 0.427 |
| XGBoost | 0.005605 | 0.009275 | 0.311 | 0.339 | 0.3781 | 0.798 | 0.427 |
| Random Forest | 0.005782 | 0.009553 | 0.269 | 0.298 | 0.3793 | 0.821 | 0.428 |
| HAR (OLS) | 0.006341 | 0.010106 | 0.182 | 0.215 | 0.4529 | 0.869 | 0.511 |
| Train mean | 0.007318 | 0.011405 | −0.042 | 0 | 0.5890 | 0.981 | 0.665 |
| Naive | 0.007262 | 0.011629 | −0.084 | −0.040 | 0.8857 | 1 | 1 |

Diebold–Mariano vs naive (negative = better than naive):

| Model | DM (SE) | p | DM (QLIKE) | p |
|---|---|---|---|---|
| Train mean | −0.17 | 0.864 | −1.58 | 0.114 |
| HAR | −1.83 | 0.067 | −4.38 | < 0.0001 |
| Linear | −2.33 | 0.020 | −4.71 | < 0.0001 |
| Ridge | −2.25 | 0.024 | −4.60 | < 0.0001 |
| Random Forest | −2.90 | 0.004 | −4.27 | < 0.0001 |
| XGBoost | −2.31 | 0.021 | −4.22 | < 0.0001 |

RMSE relative to naive by regime (n: calm 211, normal 373, turbulent 166):

| Model | Calm | Normal | Turbulent |
|---|---|---|---|
| Linear | 0.775 | 0.871 | 0.748 |
| Ridge | 0.760 | 0.888 | 0.755 |
| XGBoost | 0.732 | 0.887 | 0.793 |
| Random Forest | 0.739 | 0.908 | 0.821 |
| HAR | 0.794 | 0.975 | 0.863 |
| Train mean | 0.943 | 1.156 | 0.950 |

### 9.3 Findings
1. **Volatility is forecastable out of sample.** The best models reduce RMSE by about 23 % relative to the naive forecast (R²_oos ≈ 0.38); the improvement is statistically significant for all four 8-feature models under both losses.
2. **Model complexity did not improve accuracy.** Untuned linear regression has the lowest test RMSE; Ridge, XGBoost and Random Forest are not better. The predictable component appears to be largely linear in these features.
3. **The extended feature set adds value over HAR** (RMSE 0.771 vs 0.869 of naive), consistent with the leverage effect carried by signed returns and momentum.
4. **Loss functions rank models differently:** RMSE favours plain linear regression; QLIKE favours Ridge, XGBoost and Random Forest (≈ 0.427 vs 0.447).
5. **Gains depend on regime:** largest in calm and turbulent weeks, smallest in normal weeks. Tree models are relatively better in calm weeks, linear models in turbulent weeks.

### 9.4 Limitations
- A single chronological split; the test period is relatively calm on average. Walk-forward evaluation (Phase 2) will test robustness across periods.
- Diebold–Mariano tests with 750 overlapping forecasts are approximate (no small-sample correction applied).
- Validation scores of tuned models are optimistically biased (selected on validation); the test comparison is the unbiased one.
- Hyperparameter grids are modest; two choices lie at grid edges (D16).

## 10. Implication for Phase 2
Physics-inspired features must be added to the **same** models, splits and metrics. A claim of improvement requires a significant Diebold–Mariano test of the augmented model against the **same model without the physics features** (not against naive). Current benchmark to beat: Linear/Ridge at ≈ 0.77 of naive RMSE and ≈ 0.43 of naive QLIKE.

## References
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
