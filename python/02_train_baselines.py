"""
Step 6: Train baseline models for 5-day realized volatility.

Run from the project root, with the virtual environment active:
    python python/02_train_baselines.py

Protocol (roadmap D10-D12):
  1. Every model is fitted on TRAIN only; hyperparameters are chosen by
     validation RMSE.
  2. The chosen configuration is refitted on TRAIN + VAL.
  3. Test predictions are produced once and saved. Test metrics are NOT
     printed here: they are computed only in Step 7 (03_evaluate_models.py).

Input:
    data/processed/ml_dataset_v1.csv   (from 01_prepare_ml_data.py)
Outputs:
    results/predictions/val_predictions.csv     (models fitted on train)
    results/predictions/test_predictions.csv    (models refitted on train+val)
    results/tables/validation_results.csv       (validation leaderboard)
    results/tables/hyperparameters.json         (chosen settings)
    results/tables/har_ols_train.txt            (HAR coefficients, HAC t-stats)
    results/models/*.joblib                     (fitted models, git-ignored)
"""

from pathlib import Path
import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "ml_dataset_v1.csv"
PRED_DIR = ROOT / "results" / "predictions"
TAB_DIR = ROOT / "results" / "tables"
MODEL_DIR = ROOT / "results" / "models"
for d in (PRED_DIR, TAB_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

FEATURES = ["r", "r_lag1", "r_lag2", "abs_r", "RV5_past", "RV22_past", "sigma20", "M10"]
HAR_FEATURES = ["abs_r", "RV5_past", "RV22_past"]
TARGET = "RV5_target"
HORIZON = 5

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA, parse_dates=["Date"])
train = df[df["split"] == "train"]
val = df[df["split"] == "val"]
trval = df[df["split"].isin(["train", "val"])]
test = df[df["split"] == "test"]
print(f"Rows  train {len(train)}  val {len(val)}  train+val {len(trval)}  test {len(test)}")

y_tr, y_va, y_tv = train[TARGET].values, val[TARGET].values, trval[TARGET].values


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


val_pred = pd.DataFrame({"Date": val["Date"].dt.strftime("%Y-%m-%d"), "y_true": y_va})
test_pred = pd.DataFrame({"Date": test["Date"].dt.strftime("%Y-%m-%d"), "y_true": test[TARGET].values})
leaderboard, chosen = [], {}


def edge_note(name, grid, chosen_params):
    """Warn when a chosen hyperparameter sits at the edge of its search grid."""
    for k, values in grid.items():
        if len(values) > 2 and chosen_params[k] in (values[0], values[-1]):
            print(f"    note: {name} {k} = {chosen_params[k]} is at the edge of its grid {values}")


def record(name, p_val, p_test, params, n_configs, seconds):
    val_pred[name] = p_val
    test_pred[name] = p_test
    chosen[name] = params
    leaderboard.append({"model": name, "val_RMSE": rmse(y_va, p_val),
                        "val_MAE": mean_absolute_error(y_va, p_val),
                        "configs_tried": n_configs, "seconds": round(seconds, 1)})
    print(f"  {name:14s} val RMSE {rmse(y_va, p_val):.6f}   ({n_configs} config(s), {seconds:.1f}s)")


print("\nTraining (validation scores only; test is NOT scored here)")

# 1. Naive: next week's volatility = this week's
t0 = time.time()
record("naive", val["RV5_past"].values, test["RV5_past"].values, {}, 1, time.time() - t0)

# 2. Training mean (refit on train+val for the test forecast)
t0 = time.time()
record("train_mean", np.full(len(val), y_tr.mean()), np.full(len(test), y_tv.mean()),
       {"train_mean": float(y_tr.mean()), "trainval_mean": float(y_tv.mean())}, 1, time.time() - t0)

# 3. HAR: OLS on daily / weekly / monthly volatility
t0 = time.time()
har = LinearRegression().fit(train[HAR_FEATURES], y_tr)
har_tv = LinearRegression().fit(trval[HAR_FEATURES], y_tv)
record("HAR_OLS", har.predict(val[HAR_FEATURES]), har_tv.predict(test[HAR_FEATURES]),
       {"features": HAR_FEATURES}, 1, time.time() - t0)
joblib.dump(har_tv, MODEL_DIR / "HAR_OLS.joblib")

# Readable HAR coefficient table (train only), HAC standard errors because
# overlapping 5-day targets make the residuals autocorrelated
ols = sm.OLS(y_tr, sm.add_constant(train[HAR_FEATURES])).fit(
    cov_type="HAC", cov_kwds={"maxlags": HORIZON})
(TAB_DIR / "har_ols_train.txt").write_text(ols.summary().as_text())

# 4. Linear regression on all 8 features
t0 = time.time()
lin = LinearRegression().fit(train[FEATURES], y_tr)
lin_tv = LinearRegression().fit(trval[FEATURES], y_tv)
record("Linear_all", lin.predict(val[FEATURES]), lin_tv.predict(test[FEATURES]),
       {"features": FEATURES}, 1, time.time() - t0)
joblib.dump(lin_tv, MODEL_DIR / "Linear_all.joblib")

# 5. Ridge (standardised inputs; scaler fitted inside the pipeline on the
#    fitting data only, so no validation/test information leaks into it)
t0 = time.time()
alphas = [1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4]
scores = {}
for a in alphas:
    m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(train[FEATURES], y_tr)
    scores[a] = rmse(y_va, m.predict(val[FEATURES]))
best_a = min(scores, key=scores.get)
edge_note("Ridge", {"alpha": alphas}, {"alpha": best_a})
ridge = make_pipeline(StandardScaler(), Ridge(alpha=best_a)).fit(train[FEATURES], y_tr)
ridge_tv = make_pipeline(StandardScaler(), Ridge(alpha=best_a)).fit(trval[FEATURES], y_tv)
record("Ridge", ridge.predict(val[FEATURES]), ridge_tv.predict(test[FEATURES]),
       {"alpha": best_a}, len(alphas), time.time() - t0)
joblib.dump(ridge_tv, MODEL_DIR / "Ridge.joblib")

# 6. Random Forest
t0 = time.time()
rf_grid = {"max_depth": [3, 5, 8, None], "min_samples_leaf": [5, 20, 50], "max_features": [1.0, 0.5]}
best, best_score = None, np.inf
for md, msl, mf in itertools.product(*rf_grid.values()):
    m = RandomForestRegressor(n_estimators=500, max_depth=md, min_samples_leaf=msl,
                              max_features=mf, random_state=SEED, n_jobs=-1)
    s = rmse(y_va, m.fit(train[FEATURES], y_tr).predict(val[FEATURES]))
    if s < best_score:
        best, best_score = {"max_depth": md, "min_samples_leaf": msl, "max_features": mf}, s
edge_note("RandomForest", rf_grid, best)
rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1, **best).fit(train[FEATURES], y_tr)
rf_tv = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1, **best).fit(trval[FEATURES], y_tv)
record("RandomForest", rf.predict(val[FEATURES]), rf_tv.predict(test[FEATURES]),
       {**best, "n_estimators": 500}, int(np.prod([len(v) for v in rf_grid.values()])), time.time() - t0)
joblib.dump(rf_tv, MODEL_DIR / "RandomForest.joblib")

# 7. XGBoost (number of trees chosen by early stopping on validation)
t0 = time.time()
xgb_grid = {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1], "min_child_weight": [1, 10]}
best, best_score, best_iter = None, np.inf, None
for md, lr, mcw in itertools.product(*xgb_grid.values()):
    m = XGBRegressor(n_estimators=2000, max_depth=md, learning_rate=lr, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
                     early_stopping_rounds=50, eval_metric="rmse")
    m.fit(train[FEATURES], y_tr, eval_set=[(val[FEATURES], y_va)], verbose=False)
    s = rmse(y_va, m.predict(val[FEATURES]))
    if s < best_score:
        best, best_score, best_iter = {"max_depth": md, "learning_rate": lr, "min_child_weight": mcw}, s, m.best_iteration
edge_note("XGBoost", xgb_grid, best)
common = {"subsample": 0.8, "colsample_bytree": 0.8, "random_state": SEED, "n_jobs": -1}
xgb = XGBRegressor(n_estimators=best_iter + 1, **best, **common).fit(train[FEATURES], y_tr)
xgb_tv = XGBRegressor(n_estimators=best_iter + 1, **best, **common).fit(trval[FEATURES], y_tv)
record("XGBoost", xgb.predict(val[FEATURES]), xgb_tv.predict(test[FEATURES]),
       {**best, "n_estimators": best_iter + 1, "subsample": 0.8, "colsample_bytree": 0.8},
       int(np.prod([len(v) for v in xgb_grid.values()])), time.time() - t0)
joblib.dump(xgb_tv, MODEL_DIR / "XGBoost.joblib")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
board = pd.DataFrame(leaderboard)
board["RMSE_vs_naive"] = board["val_RMSE"] / board.loc[board.model == "naive", "val_RMSE"].iloc[0]
board = board.sort_values("val_RMSE")
board.to_csv(TAB_DIR / "validation_results.csv", index=False)
val_pred.to_csv(PRED_DIR / "val_predictions.csv", index=False)
test_pred.to_csv(PRED_DIR / "test_predictions.csv", index=False)
(TAB_DIR / "hyperparameters.json").write_text(json.dumps(chosen, indent=2, default=str))

print("\nValidation leaderboard (models fitted on train only)")
print(board.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nHAR regression on train (HAC standard errors, maxlags = 5)")
print(ols.summary().tables[1])

neg = {c: int((test_pred[c] <= 0).sum()) for c in test_pred.columns[2:]}
print("\nNon-positive test predictions per model (should be 0):", neg)
print("\nChosen hyperparameters:", json.dumps(chosen, default=str))
print("\nSaved predictions, tables and models. Test metrics are computed in Step 7.")
