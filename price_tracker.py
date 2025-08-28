""""
Daily price tracker for Yu-Gi-Oh cards in Rands

- Reads the EXACT card names from the cards.txt file
- Fetches prices via YGOPRODeck v7 in one batched call (name=Exact1|Exact2|...)
- Fetches ECB reference FX (via Frankfurter) for USD/EUR → ZAR (yesterday's rate)
- Persists rows to SQLite (prices.db), including native + ZAR prices
- Writes latest snapshot CSV (vendor columns, ZAR values)
- Prints a rich summary table

"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sqlite3
import time
from typing import Dict, List, Tuple

import pandas as pd
import requests
from rich.console import Console
from rich.table import Table

#API url for YGOPRODeck to get the card info
API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

#vendor price keys.
#Note that cardmarket is EUR, while the others are USD per the API guide
VENDORS = [
    "cardmarket_price",   # EUR
    "tcgplayer_price",    # USD
    "ebay_price",         # USD
    "amazon_price",       # USD
    "coolstuffinc_price", # USD
]

VENDOR_CCY = {
    "cardmarket_price": "EUR",
    "tcgplayer_price": "USD",
    "ebay_price": "USD",
    "amazon_price": "USD",
    "coolstuffinc_price": "USD",
}

#creates an OS-safe path to a directory meant for storing raw data
#using os.path.join is good practice for cross-platform compatibility
DB_PATH = "prices.db"
LATEST_CSV_PATH = "prices_latest.csv"
RAW_DIR = os.path.join("data", "raw")

console = Console()

#takes a file path as input (path: str) and returns a list of strings (List[str]), 
#where each string is a card name.
#returns clean list of card names from the specified file
def read_cards_file(path: str) -> List[str]:

    #If the file doesn’t exist at the provided path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Card list file not found: {path}")
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                names.append(s)
    if not names:
        raise ValueError(f"No card names found in {path}")
    return names

#ensures that the database schema is set up correctly
def ensure_schema(conn: sqlite3.Connection) -> None:

    cur = conn.cursor()
    # Price history with native + ZAR prices
    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            date            TEXT NOT NULL,      -- YYYY-MM-DD UTC
            card_id         INTEGER NOT NULL,
            card_name       TEXT NOT NULL,
            vendor          TEXT NOT NULL,      -- e.g., tcgplayer_price
            currency_native TEXT NOT NULL,      -- 'USD' or 'EUR'
            price_native    REAL,               -- nullable if API missing
            fx_rate_to_zar  REAL,               -- native -> ZAR
            fx_date         TEXT,               -- YYYY-MM-DD (ECB reference date)
            fx_source       TEXT,               -- 'Frankfurter(ECB)'
            price_zar       REAL,               -- price_native * fx_rate_to_zar
            PRIMARY KEY (date, card_id, vendor)
        )
    """)

    #cache daily FX
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fx_rates (
            fx_date TEXT NOT NULL,
            ccy     TEXT NOT NULL,   -- 'USD' or 'EUR'
            rate    REAL NOT NULL,   -- to ZAR
            source  TEXT NOT NULL,
            PRIMARY KEY (fx_date, ccy)
        )
    """)
    conn.commit()


#makes an HTTP GET request to the specified URL with optional parameters
#returns the JSON payload or raises an HTTP error if the response indicates failure
def request_with_retry(url: str, params: Dict[str, str] | None = None, max_retries: int = 3) -> dict:

    headers = {"User-Agent": "price-tracker/1.0 (+https://github.com/ShivayRam/ygo-price-tracker)"}

    backoff = 1.6

    for attempt in range(max_retries):

        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 429 and attempt < max_retries - 1:
            time.sleep(backoff**attempt)
            continue

        r.raise_for_status()
        return r.json()
    return r.json()


#takes a list of card names, combines them into a single pipe-separated string
#then it makes a request to the YGOPRODeck API to fetch price data for those cards
def fetch_prices(card_names: List[str]) -> List[dict]:

    #v7 supports pipe-separated exact names such as Baby Dragon|Time Wizard
    joined = "|".join(card_names)
    json_obj = request_with_retry(API_URL, params={"name": joined})

    if "data" not in json_obj:
        raise RuntimeError(f"Unexpected API response: {json_obj}")
    
    return json_obj["data"]

#Constructs a URL to query historical exchange rates from the Frankfurter API
#specified date, base currency, and always converts it to ZAR (South African Rand)
def frankfurter_by_date(ccy: str, fx_date: str) -> dict:

    #ECB reference rates via Frankfurter; returns {'rates': {'ZAR': ...}, 'date': 'YYYY-MM-DD', 'base': ccy}
    base_url = "https://api.frankfurter.dev"
    url = f"{base_url}/{fx_date}"
    
    return request_with_retry(url, params={"from": ccy, "to": "ZAR"})

def fetch_daily_fx(fetch_date_utc: str) -> dict:

    """Use yesterday's date to avoid today's not-yet-published ECB reference (published ~16:00 CET)."""
    d = dt.datetime.strptime(fetch_date_utc, "%Y-%m-%d").date()
    
    fx_date = (d - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    
    out = {}
    
    for ccy in ("USD", "EUR"):
        data = frankfurter_by_date(ccy, fx_date)
        out[ccy] = {
            "rate": data["rates"]["ZAR"],
            "fx_date": data["date"],
            "source": "Frankfurter(ECB)",
        }
    
    return out

def cache_raw(cards: List[dict], fetch_date: str) -> None:

    os.makedirs(RAW_DIR, exist_ok=True)

    path = os.path.join(RAW_DIR, f"{fetch_date}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": fetch_date, "data": cards}, f, ensure_ascii=False, indent=2)


def normalize_rows(cards: List[dict], fetch_date: str, fx: dict) -> List[Tuple]:

    rows: List[Tuple] = []

    for c in cards:
        card_id = c.get("id")
        card_name = c.get("name")
        prices = (c.get("card_prices") or [{}])[0]

        for vendor in VENDORS:
            raw = prices.get(vendor)
            price_native = float(raw) if (raw not in (None, "", "NaN")) else None
            ccy = VENDOR_CCY[vendor]
            rate_info = fx.get(ccy, {})
            rate = rate_info.get("rate")
            price_zar = (price_native * rate) if (price_native is not None and rate is not None) else None

            rows.append((
                fetch_date, card_id, card_name, vendor,
                ccy, price_native,
                rate, rate_info.get("fx_date"), rate_info.get("source"),
                price_zar
            ))
    return rows

def upsert_fx(conn: sqlite3.Connection, fx: dict) -> None:

    cur = conn.cursor()
    for ccy, info in fx.items():
        cur.execute("""
            INSERT OR REPLACE INTO fx_rates (fx_date, ccy, rate, source)
            VALUES (?, ?, ?, ?)
        """, (info["fx_date"], ccy, info["rate"], info["source"]))
    conn.commit()

def upsert_prices(conn: sqlite3.Connection, rows: List[Tuple]) -> int:

    cur = conn.cursor()
    cur.executemany("""
        INSERT OR REPLACE INTO price_history
            (date, card_id, card_name, vendor, currency_native, price_native,
             fx_rate_to_zar, fx_date, fx_source, price_zar)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    return cur.rowcount

def latest_snapshot(conn: sqlite3.Connection) -> pd.DataFrame:

    # Export ZAR prices in the vendor columns (wide format)
    df = pd.read_sql_query("SELECT * FROM price_history", conn)
    if df.empty:
        return df
    
    latest = df['date'].max()
    df_latest = df[df['date'] == latest]

    # Build a temporary 'value' column (ZAR) for pivoting
    temp = df_latest.assign(value=df_latest["price_zar"])
    out = (temp
           .pivot_table(index=["card_id", "card_name"], columns="vendor", values="value", aggfunc="first")
           .reset_index()
           .sort_values(["card_name", "card_id"])
           .reset_index(drop=True))
    
    return out

def pretty_print_snapshot(df: pd.DataFrame) -> None:

    if df.empty:
        console.print("[red]No data available to display.[/red]")
        return
    
    table = Table(title="Latest Yu-Gi-Oh Card Prices per vendor (ZAR)")

    for col in df.columns:
        table.add_column(col, overflow="fold")

    for _, row in df.iterrows():
        table.add_row(*[str(x if pd.notna(x) else "") for x in row.values])
        console.print(table)


def main():
    ap = argparse.ArgumentParser(description="Yu-Gi-Oh! Price Tracker (ZAR)")
    ap.add_argument("--cards", default="cards.txt", help="Path to cards.txt (one exact name per line)")
    ap.add_argument("--db", default=DB_PATH, help="SQLite DB path")
    ap.add_argument("--csv", default=LATEST_CSV_PATH, help="Output latest snapshot CSV (ZAR values)")
    ap.add_argument("--no-cache", action="store_true", help="Skip writing raw API JSON to data/raw")
    args = ap.parse_args()

    fetch_date = dt.datetime.utcnow().strftime("%Y-%m-%d")  # UTC date key

    card_names = read_cards_file(args.cards)
    cards = fetch_prices(card_names)

    if not args.no_cache:
        cache_raw(cards, fetch_date)

    conn = sqlite3.connect(args.db)
    ensure_schema(conn)

    fx = fetch_daily_fx(fetch_date)         # {'USD': {...}, 'EUR': {...}}
    upsert_fx(conn, fx)

    rows = normalize_rows(cards, fetch_date, fx)
    inserted = upsert_prices(conn, rows)

    df_latest = latest_snapshot(conn)
    df_latest.to_csv(args.csv, index=False)

    console.rule(f"[bold green]Fetched {len(cards)} cards, upserted {inserted} ZAR price rows for {fetch_date}")
    pretty_print_snapshot(df_latest)
    console.print(f"\nWrote ZAR snapshot to [bold]{args.csv}[/bold] and stored history in [bold]{args.db}[/bold].")

if __name__ == "__main__":
    main()