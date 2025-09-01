import os
import json
import pandas as pd
import yfinance as yf
from datetime import timedelta
import chardet

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EARNINGS_DATES_CSV = os.path.join(BASE_DIR, "data", "earnings_dates.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "price_data.json")

# Detect CSV encoding
with open(EARNINGS_DATES_CSV, "rb") as f:
    result = chardet.detect(f.read())
encoding = result["encoding"]

# Load earnings dates
earnings_df = pd.read_csv(EARNINGS_DATES_CSV, encoding=encoding)
earnings_df.columns = earnings_df.columns.str.strip()  # normalize column names

price_data = {}

for _, row in earnings_df.iterrows():
    ticker = row["Ticker"].strip()
    earnings_date = pd.to_datetime(row["Earnings Date"])

    start_date = (earnings_date - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (earnings_date + timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"\n📈 Fetching {ticker} from {start_date} to {end_date}...")

    # Fetch from Yahoo Finance
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        print(f"⚠ No data found for {ticker}")
        continue

    # Flatten if multi-index
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    # Reset index
    data.reset_index(inplace=True)
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")

    # Round price columns
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Format Volume
    if "Volume" in data.columns:
        data["Volume"] = data["Volume"].astype("Int64").apply(
            lambda x: f"{x:,}" if pd.notna(x) else None
        )

    # Calculate % change based on Adj Close
    data["% Change"] = (data["Adj Close"].pct_change() * 100).round(2)

    # Largest gain
    largest_gain_row = data.loc[data["% Change"].idxmax()]
    largest_gain_date = largest_gain_row["Date"]
    largest_gain_value = round(largest_gain_row["% Change"], 2)

    # Largest drop
    largest_drop_row = data.loc[data["% Change"].idxmin()]
    largest_drop_date = largest_drop_row["Date"]
    largest_drop_value = round(largest_drop_row["% Change"], 2)

    # Earnings day % change
    if earnings_date.strftime("%Y-%m-%d") in data["Date"].values:
        earnings_day_row = data.loc[data["Date"] == earnings_date.strftime("%Y-%m-%d")].iloc[0]
        earnings_day_change = round(earnings_day_row["% Change"], 2)
    else:
        earnings_day_change = None

    # Save JSON
    price_data[ticker] = {
        "summary": {
            "earnings_date": earnings_date.strftime("%Y-%m-%d"),
            "largest_gain": {
                "date": largest_gain_date,
                "percent": largest_gain_value
            },
            "largest_drop": {
                "date": largest_drop_date,
                "percent": largest_drop_value
            },
            "earnings_day_change": earnings_day_change
        },
        "data": data.to_dict(orient="records")
    }

    # Terminal summary
    print(f"🔍 Analysis for {ticker}:")
    print(f"   📅 Largest Gain: {largest_gain_date}  ({largest_gain_value:+.2f}%)")
    print(f"   📅 Largest Drop: {largest_drop_date}  ({largest_drop_value:+.2f}%)")
    if earnings_day_change is not None:
        print(f"   📊 Earnings Day ({earnings_date.strftime('%Y-%m-%d')}): {earnings_day_change:+.2f}%")
    else:
        print(f"   📊 Earnings Day ({earnings_date.strftime('%Y-%m-%d')}): N/A")

# Save file
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(price_data, f, indent=4)

print(f"\n✅ Price data saved to {OUTPUT_FILE}")
