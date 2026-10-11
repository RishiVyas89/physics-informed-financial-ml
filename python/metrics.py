"""
metrics.py - forecast evaluation tools for the volatility project.

All functions take:
    y_true : realized 5-day volatility (the target, RV5_future)
    y_pred : forecast of that volatility
as 1-D arrays of the same length, aligned on the same dates.

Contents
    mae, rmse, r2         standard error measures (on volatility)
    qlike                 QLIKE loss (Patton 2011), computed on VARIANCE = vol**2
    score_all             all four in one dictionary
    diebold_mariano       DM test (Diebold & Mariano 1995) with Newey-West (HAC)
                          variance and the Harvey-Leybourne-Newbold correction

Run `python python/metrics.py` to execute the self-tests on simulated data.
"""

import numpy as np
from scipy import stats


def _check(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Length mismatch: {y_true.size} vs {y_pred.size}")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("NaN values found - drop or align them first")
    return y_true, y_pred


def mae(y_true, y_pred):
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    y_true, y_pred = _check(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def qlike_series(y_true, y_pred):
    """Per-day QLIKE loss on variances. Zero for a perfect forecast, always >= 0."""
    y_true, y_pred = _check(y_true, y_pred)
    if (y_pred <= 0).any():
        raise ValueError("QLIKE needs strictly positive forecasts")
    if (y_true <= 0).any():
        raise ValueError("QLIKE needs strictly positive realized volatility")
    ratio = (y_true ** 2) / (y_pred ** 2)
    return ratio - np.log(ratio) - 1.0


def qlike(y_true, y_pred):
    return float(np.mean(qlike_series(y_true, y_pred)))


def score_all(y_true, y_pred):
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "QLIKE": qlike(y_true, y_pred),
    }


def diebold_mariano(y_true, pred_a, pred_b, loss="QLIKE", horizon=5):
    """
    Tests whether forecast A and forecast B have equal expected loss.

    d_t = loss(A) - loss(B).  Negative DM statistic => A is better.
    The 5-day targets overlap, so d_t is autocorrelated; Newey-West
    uses horizon-1 = 4 lags. Returns (DM statistic, two-sided p-value).
    """
    y_true, pred_a = _check(y_true, pred_a)
    _, pred_b = _check(y_true, pred_b)
    if loss == "QLIKE":
        d = qlike_series(y_true, pred_a) - qlike_series(y_true, pred_b)
    elif loss == "MSE":
        d = (y_true - pred_a) ** 2 - (y_true - pred_b) ** 2
    elif loss == "MAE":
        d = np.abs(y_true - pred_a) - np.abs(y_true - pred_b)
    else:
        raise ValueError("loss must be 'QLIKE', 'MSE' or 'MAE'")

    n = d.size
    d_bar = d.mean()
    dc = d - d_bar
    lags = horizon - 1
    long_run_var = np.sum(dc * dc) / n
    for k in range(1, lags + 1):
        weight = 1.0 - k / (lags + 1)  # Bartlett kernel
        long_run_var += 2.0 * weight * np.sum(dc[k:] * dc[:-k]) / n
    if long_run_var <= 0:
        raise ValueError("Non-positive long-run variance; forecasts may be identical")

    dm = d_bar / np.sqrt(long_run_var / n)
    # Harvey, Leybourne & Newbold (1997) small-sample correction
    h = horizon
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p_value = 2.0 * stats.t.sf(np.abs(dm), df=n - 1)
    return float(dm), float(p_value)


# --------------------------------------------------------------------------
# Self-tests on simulated data (no real project data used)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1000
    passed = 0

    # A positive, clustered "true volatility" series
    log_vol = np.zeros(n)
    for t in range(1, n):
        log_vol[t] = 0.95 * log_vol[t - 1] + 0.2 * rng.standard_normal()
    y = 0.01 * np.exp(log_vol)

    # 1. A perfect forecast scores zero error, R2 = 1, QLIKE = 0
    s = score_all(y, y)
    assert s["MAE"] == 0 and s["RMSE"] == 0 and abs(s["R2"] - 1) < 1e-12
    assert abs(s["QLIKE"]) < 1e-12
    print("Test 1 passed: perfect forecast -> MAE=RMSE=QLIKE=0, R2=1"); passed += 1

    # 2. Hand-checkable numbers
    yt, yp = np.array([1.0, 2.0]), np.array([2.0, 2.0])
    assert abs(mae(yt, yp) - 0.5) < 1e-12
    assert abs(rmse(yt, yp) - np.sqrt(0.5)) < 1e-12
    expected_q = ((0.25 - np.log(0.25) - 1) + 0) / 2
    assert abs(qlike(yt, yp) - expected_q) < 1e-12
    print(f"Test 2 passed: hand-checked MAE=0.5, RMSE={np.sqrt(0.5):.4f}, QLIKE={expected_q:.4f}"); passed += 1

    # 3. QLIKE is never negative, and punishes under-prediction more than over-prediction
    q = qlike_series(y, y * rng.uniform(0.5, 1.5, n))
    assert (q >= -1e-12).all()
    assert qlike(y, 0.5 * y) > qlike(y, 1.5 * y)
    print("Test 3 passed: QLIKE >= 0 and under-prediction costs more"); passed += 1

    # 4. DM test detects a clearly better forecast...
    good = y * np.exp(0.05 * rng.standard_normal(n))
    bad = y * np.exp(0.40 * rng.standard_normal(n))
    dm, p = diebold_mariano(y, good, bad)
    assert dm < 0 and p < 0.01
    print(f"Test 4 passed: good vs bad -> DM={dm:.2f}, p={p:.2g} (good wins)"); passed += 1

    # 5. ...and does NOT find a difference between two equally good forecasts
    rejections = 0
    for _ in range(200):
        a = y * np.exp(0.2 * rng.standard_normal(n))
        b = y * np.exp(0.2 * rng.standard_normal(n))
        if diebold_mariano(y, a, b)[1] < 0.05:
            rejections += 1
    rate = rejections / 200
    assert rate < 0.10
    print(f"Test 5 passed: equal forecasts -> false alarm rate {rate:.1%} (should be near 5%)"); passed += 1

    # 6. Bad inputs are rejected
    bad_cases = [
        ([1.0], [1.0, 2.0]),        # length mismatch
        ([1.0, 1.0], [1.0, -1.0]),  # negative forecast
        ([1.0, 1.0], [1.0, np.nan]),  # missing value
    ]
    for yt_bad, yp_bad in bad_cases:
        try:
            qlike(yt_bad, yp_bad)
            raise AssertionError("bad input was accepted")
        except ValueError:
            pass
    print("Test 6 passed: length mismatch, negative forecast and NaN all rejected"); passed += 1

    print(f"\nAll {passed} tests passed.")
