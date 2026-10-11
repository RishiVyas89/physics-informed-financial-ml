r"""
quick_baseline_har.py - PRELIMINARY baseline: naive vs HAR on the validation set.

Built by Person B for the one-month progress presentation.
Reads the raw price file directly, rebuilds the HAR features in Python,
makes purged chronological splits (65 / 15 / 20), fits HAR on TRAIN only,
and scores on VALIDATION only. The TEST set is never touched here.

Usage (from the project folder, with .venv active):
    python python\quick_baseline_har.py

Outputs:
    results/preliminary_scores.csv
    results/preliminary_har_vs_naive.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import score_all, diebold_mariano  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "SP500_2011_2026.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
HORIZON = 5

# ---------------------------------------------------------------- 1. data
prices = pd.read_csv(RAW, parse_dates=["Date"]).set_index("Date").sort_index()
print(f"Raw prices: {len(prices)} rows, {prices.index[0].date()} to {prices.index[-1].date()}")

r = np.log(prices["Close"]).diff().dropna()        # daily log returns
r2 = r ** 2

# ---------------------------------------------------------------- 2. features (past only)
df = pd.DataFrame(index=r.index)
df["abs_r"] = r.abs()                                          # daily
df["RV5_past"] = np.sqrt(r2.rolling(5).sum())                  # weekly
df["RV22_past"] = np.sqrt(r2.rolling(22).sum() * 5 / 22)       # monthly, rescaled to 5-day units

# ---------------------------------------------------------------- 3. target (future)
# RV5_future at day t = sqrt(sum of squared returns on days t+1 ... t+5)
df["RV5_future"] = np.sqrt(r2.rolling(5).sum().shift(-HORIZON))
df = df.dropna()
print(f"Feature table: {len(df)} rows (Mathematica features_v1.csv has 3,745)")

# Leakage sanity check: the target must equal RV5_past shifted 5 days back
chk = (df["RV5_future"] - df["RV5_past"].shift(-HORIZON)).dropna().abs().max()
print(f"Target alignment check, max difference: {chk:.2e} (should be ~0)")

# ---------------------------------------------------------------- 4. purged chronological splits
n = len(df)
i_val, i_test = int(0.65 * n), int(0.80 * n)
train = df.iloc[: i_val - HORIZON]          # purge last 5 rows: targets reach into validation
val = df.iloc[i_val : i_test - HORIZON]     # purge last 5 rows: targets reach into test
# test = df.iloc[i_test:]  <- deliberately NOT created. Opened once, at the very end of the project.

for name, part in (("Train", train), ("Validation", val)):
    print(f"{name:<11}{len(part):>5} rows  {part.index[0].date()} to {part.index[-1].date()}")
print(f"Test       {n - i_test:>5} rows  (held out, untouched)")

# ---------------------------------------------------------------- 5. models
X_cols = ["abs_r", "RV5_past", "RV22_past"]
y_val = val["RV5_future"].values

pred = {}
pred["Training mean"] = np.full(len(val), train["RV5_future"].mean())
pred["Naive (RV5_past)"] = val["RV5_past"].values

har = LinearRegression().fit(train[X_cols], train["RV5_future"])
har_pred = har.predict(val[X_cols])
floor = train["RV5_future"].min()
n_clipped = int((har_pred < floor).sum())
pred["HAR"] = np.maximum(har_pred, floor)    # QLIKE needs positive forecasts

print("\nHAR coefficients (fitted on train only):")
print(f"  intercept   {har.intercept_: .5f}")
for c, b in zip(X_cols, har.coef_):
    print(f"  {c:<10}  {b: .4f}")
print(f"  forecasts raised to floor: {n_clipped}")

# ---------------------------------------------------------------- 6. scores
scores = pd.DataFrame({m: score_all(y_val, p) for m, p in pred.items()}).T
print("\nValidation scores (lower is better, except R2):")
print(scores.round(5).to_string())
scores.to_csv(OUT / "preliminary_scores.csv")

print("\nDiebold-Mariano, HAR vs Naive (negative = HAR better):")
for loss in ("QLIKE", "MSE"):
    dm, p = diebold_mariano(y_val, pred["HAR"], pred["Naive (RV5_past)"], loss=loss)
    verdict = "significant" if p < 0.05 else "not significant"
    print(f"  {loss:<6} DM = {dm:6.2f}   p = {p:.4f}   ({verdict} at 5%)")

# ---------------------------------------------------------------- 7. figure
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(val.index, y_val, color="black", lw=1.2, label="Actual next-5-day volatility")
ax.plot(val.index, pred["Naive (RV5_past)"], color="tab:orange", lw=0.9, alpha=0.8, label="Naive")
ax.plot(val.index, pred["HAR"], color="tab:blue", lw=1.2, label="HAR")
ax.set_title("Preliminary: HAR vs naive on the validation period")
ax.set_ylabel("5-day realized volatility")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "preliminary_har_vs_naive.png", dpi=150)
print(f"\nSaved results/preliminary_scores.csv and results/preliminary_har_vs_naive.png")
