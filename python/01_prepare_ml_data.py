"""
Step 5: Load features_v1.csv, verify it independently in Python, and build
purged chronological train / validation / test splits.

Run from the project root, with the virtual environment active:
    python python/01_prepare_ml_data.py

Inputs (not in Git; regenerate with download_sp500.py and notebook 02):
    data/raw/SP500_2011_2026.csv
    data/processed/features_v1.csv

Outputs:
    data/processed/ml_dataset_v1.csv        features + target + 'split' column (git-ignored)
    results/tables/split_definition.csv     split dates and row counts (committed)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "SP500_2011_2026.csv"
FEAT_FILE = ROOT / "data" / "processed" / "features_v1.csv"
OUT_DATA = ROOT / "data" / "processed" / "ml_dataset_v1.csv"
OUT_SPLITS = ROOT / "results" / "tables" / "split_definition.csv"

FEATURES = ["r", "r_lag1", "r_lag2", "abs_r", "RV5_past", "RV22_past", "sigma20", "M10"]
TARGET = "RV5_target"
ALL_COLS = ["Date", "Close"] + FEATURES + [TARGET]

# Known values from Mathematica notebook 02 (docs/feature_dictionary.md)
EXPECTED_ROWS = 3745
EXPECTED_FIRST_DATE = "2011-05-04"
EXPECTED_LAST_DATE = "2026-03-25"
EXPECTED_LAST_ROW = {"Close": 6591.9, "RV5_past": 0.0203277, TARGET: 0.0385215}

# Split proportions (roadmap D10) and purge length = target horizon
TRAIN_FRAC, VAL_FRAC = 0.65, 0.15
HORIZON = 5

failures = []


def check(name, ok, detail=""):
    """Print PASS/FAIL for one check and remember failures."""
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------------------
# 1. Load the feature table
# ---------------------------------------------------------------------------
print("\n1. Loading features_v1.csv")
df = pd.read_csv(FEAT_FILE, parse_dates=["Date"])
# Mathematica may write very small numbers as e.g. 3.4*^-6; convert to 3.4e-6
for col in df.columns[1:]:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace("*^", "e", regex=False))
print(f"   shape = {df.shape}")

check("columns match feature dictionary", list(df.columns) == ALL_COLS, str(list(df.columns)))
check("row count", len(df) == EXPECTED_ROWS, f"{len(df)} rows")
check("first date", str(df["Date"].iloc[0].date()) == EXPECTED_FIRST_DATE, str(df["Date"].iloc[0].date()))
check("last date", str(df["Date"].iloc[-1].date()) == EXPECTED_LAST_DATE, str(df["Date"].iloc[-1].date()))
check("no missing values", int(df.isna().sum().sum()) == 0, f"{int(df.isna().sum().sum())} missing")
check("dates strictly increasing", bool(df["Date"].is_monotonic_increasing and df["Date"].is_unique))
for col, val in EXPECTED_LAST_ROW.items():
    got = df[col].iloc[-1]
    check(f"last-row {col} matches Mathematica", abs(got - val) < 1e-6, f"{got:.7g} vs {val}")

# Target identity: target in row k == RV5_past in row k+5
diff = np.abs(df[TARGET].values[:-HORIZON] - df["RV5_past"].values[HORIZON:]).max()
check("target identity RV5_target[k] == RV5_past[k+5]", diff < 1e-15, f"max |diff| = {diff:.3g}")

# ---------------------------------------------------------------------------
# 2. Independent re-computation in Python from the raw prices
# ---------------------------------------------------------------------------
print("\n2. Re-computing every feature and the target from raw prices (pandas)")
raw = pd.read_csv(RAW_FILE, parse_dates=["Date"])
close = raw.set_index("Date")["Close"]
r = np.log(close).diff()
r2 = r ** 2

py = pd.DataFrame({
    "Close": close,
    "r": r,
    "r_lag1": r.shift(1),
    "r_lag2": r.shift(2),
    "abs_r": r.abs(),
    "RV5_past": np.sqrt(r2.rolling(5).sum()),
    "RV22_past": np.sqrt(r2.rolling(22).sum()),
    "sigma20": r.rolling(20).std(ddof=1),
    "M10": np.log(close / close.shift(10)),
    TARGET: np.sqrt(r2.rolling(HORIZON).sum().shift(-HORIZON)),
})
py = py.loc[df["Date"]]  # same dates as the Mathematica table

for col in ["Close"] + FEATURES + [TARGET]:
    d = np.abs(py[col].values - df[col].values).max()
    check(f"{col:10s} Python == Mathematica", d < 1e-12, f"max |diff| = {d:.2g}")

# ---------------------------------------------------------------------------
# 3. Chronological split with purging
# ---------------------------------------------------------------------------
print("\n3. Chronological 65 / 15 / 20 split with purging")
n = len(df)
n_train_raw = int(np.floor(TRAIN_FRAC * n))
n_val_raw = int(np.floor(VAL_FRAC * n))

split = np.empty(n, dtype=object)
split[:n_train_raw] = "train"
split[n_train_raw:n_train_raw + n_val_raw] = "val"
split[n_train_raw + n_val_raw:] = "test"

# Purge: the last HORIZON rows of train and of val have targets that use
# returns from the next period, so they are removed from modelling.
split[n_train_raw - HORIZON:n_train_raw] = "purged"
split[n_train_raw + n_val_raw - HORIZON:n_train_raw + n_val_raw] = "purged"
df["split"] = split

# Check: no target window crosses into a later split
date_pos = {d: i for i, d in enumerate(df["Date"])}
order = {"train": 0, "val": 1, "test": 2}
crossing = 0
for name in ["train", "val"]:
    idx = np.where(df["split"].values == name)[0]
    last = idx[-1]
    # target of the last kept row uses rows last+1 .. last+HORIZON
    future_splits = set(df["split"].values[last + 1:last + 1 + HORIZON]) - {"purged"}
    if any(order[s] > order[name] for s in future_splits):
        crossing += 1
check("no target window crosses a split boundary", crossing == 0)
check("splits are in time order",
      df.loc[df.split == "train", "Date"].max() < df.loc[df.split == "val", "Date"].min()
      < df.loc[df.split == "val", "Date"].max() < df.loc[df.split == "test", "Date"].min())

rows = []
for name in ["train", "purged", "val", "test"]:
    part = df[df["split"] == name]
    if name == "purged":
        rows.append({"split": name, "rows": len(part),
                     "start": ", ".join(str(d.date()) for d in part["Date"].iloc[[0, HORIZON]]),
                     "end": "", "target_mean": np.nan, "target_sd": np.nan, "target_max": np.nan})
        continue
    rows.append({"split": name, "rows": len(part),
                 "start": str(part["Date"].iloc[0].date()), "end": str(part["Date"].iloc[-1].date()),
                 "target_mean": part[TARGET].mean(), "target_sd": part[TARGET].std(),
                 "target_max": part[TARGET].max()})
summary = pd.DataFrame(rows)
check("all rows accounted for", summary["rows"].sum() == n, f"{summary['rows'].sum()} of {n}")

pd.set_option("display.width", 120)
print()
print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

# ---------------------------------------------------------------------------
# 4. Save outputs
# ---------------------------------------------------------------------------
print("\n4. Saving")
if failures:
    print(f"\n{len(failures)} check(s) FAILED - nothing saved:")
    for f in failures:
        print("   -", f)
    sys.exit(1)

OUT_SPLITS.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_DATA, index=False, date_format="%Y-%m-%d")
summary.to_csv(OUT_SPLITS, index=False)
print(f"   {OUT_DATA.relative_to(ROOT)}")
print(f"   {OUT_SPLITS.relative_to(ROOT)}")
print("\nAll checks passed.")
