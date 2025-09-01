import json
import os
from datetime import datetime, timedelta

# ---------- Utility Functions ----------

def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_percent_change(open_price, close_price):
    if open_price is not None and close_price is not None and open_price != 0:
        return ((close_price - open_price) / open_price) * 100
    return None

def find_price_on_date(prices, target_date):
    """Find the price entry for a specific date string (YYYY-MM-DD)."""
    for p in prices:
        if p["Date"] == target_date:
            return p
    return None

def calculate_change(prices, earnings_date, days):
    """% change from N days before/after earnings_date to earnings_date."""
    earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
    compare_dt = earnings_dt + timedelta(days=days)
    earnings_entry = find_price_on_date(prices, earnings_date)
    compare_entry = find_price_on_date(prices, compare_dt.strftime("%Y-%m-%d"))

    if earnings_entry and compare_entry:
        return calculate_percent_change(compare_entry["Close"], earnings_entry["Close"])
    return None

def calculate_volatility(prices, earnings_date, days=10, pre=True):
    """Calculate standard deviation of % changes in the given window."""
    from statistics import pstdev

    earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
    if pre:
        window_dates = [(earnings_dt - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days+1)]
    else:
        window_dates = [(earnings_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days+1)]

    pct_changes = []
    for date_str in window_dates:
        entry = find_price_on_date(prices, date_str)
        if entry:
            change = calculate_percent_change(entry["Open"], entry["Close"])
            if change is not None:
                pct_changes.append(change)

    return round(pstdev(pct_changes), 2) if pct_changes else None

def calculate_gap_vs_intraday(prices, earnings_date):
    """Compare earnings day gap vs intraday move."""
    earnings_entry = find_price_on_date(prices, earnings_date)
    if not earnings_entry:
        return None

    # Previous Close can be missing, fallback to previous day's Close
    gap_change = None
    if "Previous Close" in earnings_entry:
        gap_change = calculate_percent_change(earnings_entry["Previous Close"], earnings_entry["Open"])
    else:
        # Try to get previous day's close
        earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
        prev_date_str = (earnings_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        prev_entry = find_price_on_date(prices, prev_date_str)
        if prev_entry:
            gap_change = calculate_percent_change(prev_entry["Close"], earnings_entry["Open"])

    intraday_change = calculate_percent_change(earnings_entry["Open"], earnings_entry["Close"])

    if gap_change is not None and intraday_change is not None:
        return round(gap_change - intraday_change, 2)
    return None

def calculate_earnings_day_change(prices, earnings_date):
    """% change from open to close on earnings day."""
    earnings_entry = find_price_on_date(prices, earnings_date)
    if earnings_entry:
        return calculate_percent_change(earnings_entry["Open"], earnings_entry["Close"])
    return None


# ---------- Main Script ----------

def main():
    transcripts_path = "data/parsed_transcripts.json"
    prices_path = "data/price_data.json"
    output_path = "data/analyzed_transcripts.json"

    transcripts = load_json(transcripts_path)
    price_data = load_json(prices_path)

    if not isinstance(price_data, dict):
        raise ValueError("⚠ data/price_data.json must be a dictionary keyed by ticker.")

    analyzed_transcripts = []

    for transcript in transcripts:
        ticker = transcript.get("ticker")
        if not ticker:
            continue

        ticker_info = price_data.get(ticker, {})
        prices = ticker_info.get("data", [])
        earnings_date = ticker_info.get("summary", {}).get("earnings_date")

        if not prices or not earnings_date:
            print(f"⚠ Missing prices or earnings date for {ticker}")
            continue

        # Get largest gain/drop info from summary if available
        largest_gain_summary = ticker_info.get("summary", {}).get("largest_gain")
        largest_drop_summary = ticker_info.get("summary", {}).get("largest_drop")

        # Prepare largest_gain dict with date and percent
        if largest_gain_summary:
            # Defensive: summary may lack Open/Close, so fallback to percent directly if exists
            largest_gain_date = largest_gain_summary.get("date") or largest_gain_summary.get("Date")
            largest_gain_percent = largest_gain_summary.get("percent")
            if largest_gain_percent is None and "Open" in largest_gain_summary and "Close" in largest_gain_summary:
                largest_gain_percent = calculate_percent_change(largest_gain_summary["Open"], largest_gain_summary["Close"])
            largest_gain = {
                "date": largest_gain_date,
                "percent": largest_gain_percent
            }
        else:
            largest_gain = None

        # Same for largest_drop
        if largest_drop_summary:
            largest_drop_date = largest_drop_summary.get("date") or largest_drop_summary.get("Date")
            largest_drop_percent = largest_drop_summary.get("percent")
            if largest_drop_percent is None and "Open" in largest_drop_summary and "Close" in largest_drop_summary:
                largest_drop_percent = calculate_percent_change(largest_drop_summary["Open"], largest_drop_summary["Close"])
            else:
                largest_drop_percent = largest_drop_summary.get("percent")
            largest_drop = {
                "date": largest_drop_date,
                "percent": largest_drop_percent
            }
        else:
            largest_drop = None

        # Calculate metrics
        earnings_day_change = calculate_earnings_day_change(prices, earnings_date)
        pre_earnings_change = calculate_change(prices, earnings_date, days=-5)
        post_earnings_change = calculate_change(prices, earnings_date, days=5)
        volatility_pre = calculate_volatility(prices, earnings_date, days=10, pre=True)
        volatility_post = calculate_volatility(prices, earnings_date, days=10, pre=False)
        gap_vs_intraday = calculate_gap_vs_intraday(prices, earnings_date)

        enriched = {
            **transcript,
            "earnings_day_change": round(earnings_day_change, 2) if earnings_day_change is not None else None,
            "pre_earnings_change_5d": round(pre_earnings_change, 2) if pre_earnings_change is not None else None,
            "post_earnings_change_5d": round(post_earnings_change, 2) if post_earnings_change is not None else None,
            "largest_gain": {
                "date": largest_gain["date"],
                "percent": round(largest_gain["percent"], 2) if largest_gain["percent"] is not None else None
            } if largest_gain else None,
            "largest_drop": {
                "date": largest_drop["date"],
                "percent": round(largest_drop["percent"], 2) if largest_drop["percent"] is not None else None
            } if largest_drop else None,
            "volatility_pre_10d": volatility_pre,
            "volatility_post_10d": volatility_post,
            "gap_vs_intraday": gap_vs_intraday
        }

        analyzed_transcripts.append(enriched)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analyzed_transcripts, f, indent=4, ensure_ascii=False)

    print(f"✅ Analysis complete. Saved to {output_path}")


if __name__ == "__main__":
    main()
