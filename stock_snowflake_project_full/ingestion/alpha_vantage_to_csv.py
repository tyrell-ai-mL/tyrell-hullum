import requests
import csv
import os
from datetime import datetime

# Simple script to pull daily stock prices from Alpha Vantage and save to CSV.
# You must sign up at https://www.alphavantage.co and get a free API key.
#
# Then run:
#   export ALPHAVANTAGE_API_KEY='your_key_here'
#   python alpha_vantage_to_csv.py
#
# The script will create data/sample_prices.csv which you can upload to Snowflake.

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA"]

BASE_URL = "https://www.alphavantage.co/query"

os.makedirs("data", exist_ok=True)
output_file = os.path.join("data", "sample_prices.csv")

fieldnames = [
    "symbol",
    "trade_date",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume"
]

if not API_KEY:
    raise RuntimeError("Please set the ALPHAVANTAGE_API_KEY environment variable.")

def fetch_symbol_daily(symbol: str):
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    ts = data.get("Time Series (Daily)", {})
    rows = []
    for date_str, values in ts.items():
        rows.append({
            "symbol": symbol,
            "trade_date": date_str,
            "open_price": values.get("1. open"),
            "high_price": values.get("2. high"),
            "low_price": values.get("3. low"),
            "close_price": values.get("4. close"),
            "volume": values.get("6. volume"),
        })
    return rows

all_rows = []
for sym in SYMBOLS:
    print(f"Fetching data for {sym}...")
    all_rows.extend(fetch_symbol_daily(sym))

# Sort by symbol + date
all_rows.sort(key=lambda r: (r["symbol"], r["trade_date"]))

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Wrote {len(all_rows)} rows to {output_file}")
