"""
Download 15 years of daily S&P 500 (^GSPC) data from Yahoo Finance
and save it as both CSV and Excel in the same folder as this script.

Setup (run once in a terminal / Command Prompt):
    pip install yfinance pandas openpyxl

Run:
    python download_sp500.py
"""

from pathlib import Path

import yfinance as yf

TICKER = "^GSPC"
START = "2011-04-01"
END = "2026-04-02"  # end date is exclusive, so this includes Apr 1, 2026

out_dir = Path(__file__).resolve().parent

print(f"Downloading {TICKER} from {START} to {END} ...")
df = yf.download(TICKER, start=START, end=END, auto_adjust=False, progress=False)

if df.empty:
    raise SystemExit("No data returned. Check your internet connection and try again.")

# yfinance returns two-level column headers (e.g. ('Close', '^GSPC')); flatten them
if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
    df.columns = df.columns.get_level_values(0)

df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
df.index = df.index.date
df.index.name = "Date"
df = df.round(2)

csv_path = out_dir / "SP500_2011_2026.csv"
xlsx_path = out_dir / "SP500_2011_2026.xlsx"

df.to_csv(csv_path)
df.to_excel(xlsx_path, sheet_name="SP500 Daily")

print(f"Rows downloaded: {len(df)}")
print(f"First date: {df.index[0]}   Last date: {df.index[-1]}")
print(f"Saved: {csv_path}")
print(f"Saved: {xlsx_path}")
