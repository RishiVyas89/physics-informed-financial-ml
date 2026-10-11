"""
Step 7: Evaluate all baseline models on the TEST set (done once).

Run from the project root, with the virtual environment active:
    python python/03_evaluate_models.py

Inputs:
    results/predictions/test_predictions.csv   (from 02_train_baselines.py)
    data/processed/ml_dataset_v1.csv           (for RV5_past regimes)
Outputs:
    results/metrics/test_metrics.csv           MAE, RMSE, R2, R2_oos, QLIKE per model
    results/metrics/dm_tests.csv               Diebold-Mariano tests vs naive
    results/metrics/regime_metrics.csv         errors in calm / normal / turbulent weeks
    results/figures/09_test_actual_vs_predicted.png
    results/figures/10_test_scatter.png

Definitions (roadmap D12):
    y      realized 5-day volatility (target);  f = forecast
    QLIKE  = y^2/f^2 - ln(y^2/f^2) - 1   (Patton 2011, on variances; 0 = perfect)
    R2     = 1 - SSE / sum((y - mean_test(y))^2)
    R2_oos = 1 - SSE / SSE(train+val mean forecast)   (benchmark known in advance)
    DM     = mean(d) / sqrt(HAC variance / T),  d_t = loss(model)_t - loss(naive)_t,
             Newey-West lags = 5 (overlapping 5-day targets). Negative DM = model better.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "predictions" / "test_predictions.csv"
DATA = ROOT / "data" / "processed" / "ml_dataset_v1.csv"
MET_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "results" / "figures"
MET_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

HAC_LAGS = 5
BENCH = "naive"

pred = pd.read_csv(PRED, parse_dates=["Date"])
data = pd.read_csv(DATA, parse_dates=["Date"])
models = [c for c in pred.columns if c not in ("Date", "y_true")]
y = pred["y_true"].values
T = len(y)
print(f"Test set: {T} forecasts, {pred.Date.iloc[0].date()} to {pred.Date.iloc[-1].date()}")
print(f"Models: {models}")

# Benchmark mean known before the test period (train+val), for R2_oos
trval_mean = data.loc[data.split.isin(["train", "val"]), "RV5_target"].mean()
check = np.abs(pred["train_mean"].values - trval_mean).max()
print(f"Check: train_mean forecast equals train+val mean  (max |diff| = {check:.2g})")

# Forecasts must be positive for QLIKE; floor tiny/negative values and report
FLOOR = 1e-6
n_floored = {m: int((pred[m] < FLOOR).sum()) for m in models}
if any(n_floored.values()):
    print("Note: forecasts floored at 1e-6 for QLIKE:", n_floored)


def losses(y, f):
    f = np.maximum(f, FLOOR)
    ratio = y**2 / f**2
    return {"SE": (y - f) ** 2, "AE": np.abs(y - f), "QLIKE": ratio - np.log(ratio) - 1}


def dm_test(d, lags=HAC_LAGS):
    """Diebold-Mariano: regress loss differential on a constant with HAC errors."""
    res = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(res.tvalues[0])
    return t, float(2 * (1 - stats.norm.cdf(abs(t))))


# ---------------------------------------------------------------------------
# 1. Overall test metrics
# ---------------------------------------------------------------------------
L = {m: losses(y, pred[m].values) for m in models}
sse_bench = ((y - trval_mean) ** 2).sum()
rows = []
for m in models:
    sse = L[m]["SE"].sum()
    rows.append({"model": m,
                 "MAE": L[m]["AE"].mean(),
                 "RMSE": np.sqrt(L[m]["SE"].mean()),
                 "R2": 1 - sse / ((y - y.mean()) ** 2).sum(),
                 "R2_oos": 1 - sse / sse_bench,
                 "QLIKE": L[m]["QLIKE"].mean()})
met = pd.DataFrame(rows)
met["RMSE_vs_naive"] = met["RMSE"] / met.loc[met.model == BENCH, "RMSE"].iloc[0]
met["QLIKE_vs_naive"] = met["QLIKE"] / met.loc[met.model == BENCH, "QLIKE"].iloc[0]
met = met.sort_values("RMSE")
met.to_csv(MET_DIR / "test_metrics.csv", index=False)

pd.set_option("display.width", 140)
print("\n1. TEST METRICS (sorted by RMSE)")
print(met.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# ---------------------------------------------------------------------------
# 2. Diebold-Mariano tests vs naive
# ---------------------------------------------------------------------------
dm_rows = []
for m in models:
    if m == BENCH:
        continue
    row = {"model": m}
    for loss in ("SE", "QLIKE"):
        t, p = dm_test(L[m][loss] - L[BENCH][loss])
        row[f"DM_{loss}"] = t
        row[f"p_{loss}"] = p
    dm_rows.append(row)
dm = pd.DataFrame(dm_rows)
dm.to_csv(MET_DIR / "dm_tests.csv", index=False)
print(f"\n2. DIEBOLD-MARIANO vs naive (HAC lags = {HAC_LAGS}; negative = better than naive)")
print(dm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------------------------------------------------------------------------
# 3. Calm / normal / turbulent weeks
#    Regime = tercile of RV5_past (known at forecast time), with cut-offs taken
#    from train+val so the test period does not define its own thresholds.
# ---------------------------------------------------------------------------
cuts = data.loc[data.split.isin(["train", "val"]), "RV5_past"].quantile([1 / 3, 2 / 3]).values
rv_past = data.set_index("Date").loc[pred["Date"], "RV5_past"].values
regime = np.where(rv_past <= cuts[0], "calm", np.where(rv_past <= cuts[1], "normal", "turbulent"))
reg_rows = []
for g in ["calm", "normal", "turbulent"]:
    mask = regime == g
    for m in models:
        reg_rows.append({"regime": g, "n": int(mask.sum()), "model": m,
                         "RMSE": np.sqrt(L[m]["SE"][mask].mean()),
                         "QLIKE": L[m]["QLIKE"][mask].mean()})
reg = pd.DataFrame(reg_rows)
reg["RMSE_vs_naive"] = reg["RMSE"] / reg.groupby("regime")["RMSE"].transform(
    lambda s: s[reg.loc[s.index, "model"] == BENCH].iloc[0])
reg.to_csv(MET_DIR / "regime_metrics.csv", index=False)
print(f"\n3. BY REGIME (RV5_past terciles from train+val: {cuts[0]:.5f}, {cuts[1]:.5f})")
print(reg.pivot(index="model", columns="regime", values="RMSE_vs_naive")
      .loc[met.model].to_string(float_format=lambda x: f"{x:.3f}"))
print("   (values = RMSE relative to naive within each regime; < 1 is better)")
print("   counts:", {g: int((regime == g).sum()) for g in ["calm", "normal", "turbulent"]})

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
best_ml = met[~met.model.isin([BENCH, "train_mean", "HAR_OLS"])].iloc[0]["model"]
show = [BENCH, "HAR_OLS", best_ml]

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(pred.Date, y, color="black", lw=1.2, label="Actual RV (next 5 days)")
for m, c in zip(show, ["tab:gray", "tab:blue", "tab:red"]):
    ax.plot(pred.Date, pred[m], lw=1, color=c, alpha=0.85, label=m)
ax.set_title("Test period: actual vs predicted 5-day realized volatility")
ax.set_xlabel("Forecast date")
ax.set_ylabel("RV (5-day, not annualized)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "09_test_actual_vs_predicted.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
lim = [0, max(y.max(), pred[show].values.max()) * 1.05]
for ax, m in zip(axes, show):
    ax.scatter(pred[m], y, s=6, alpha=0.5)
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_title(f"{m}  (RMSE {np.sqrt(L[m]['SE'].mean()):.5f})")
    ax.set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
fig.suptitle("Test period: predicted vs actual (dashed = perfect forecast)")
fig.tight_layout()
fig.savefig(FIG_DIR / "10_test_scatter.png", dpi=150)

print(f"\n4. Figures saved (showing {show}). Metrics saved to results/metrics/.")
