#!/usr/bin/env python3
"""
Daily price tracker for Yu-Gi-Oh cards in Rands (ZAR), with rarity-aware averages.

Outputs a single combined 'latest' CSV table with columns:
card_id, card_name, rarity, sets_count,
amazon_price, cardmarket_price, coolstuffinc_price, ebay_price, tcgplayer_price,
avg_set_price_usd, avg_set_price_zar
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

# ----------------------------
# Config & constants
# ----------------------------
API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"  # supports pipe-separated exact names
# YGOPRODeck: card_prices has vendor keys; card_sets entries carry set_rarity + set_price (USD).  # docs
# Vendors: Cardmarket (EUR), others (USD). We'll convert all vendors -> ZAR.                     # docs

VENDORS = [
    "cardmarket_price",   # EUR (→ ZAR)
    "tcgplayer_price",    # USD (→ ZAR)
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

DB_PATH = "prices.db"
LATEST_CSV_PATH = "prices_latest.csv"   # combined output you asked for
RARITY_CSV_PATH = "rarity_latest.csv"   # helper/export (optional to keep)
RAW_DIR = os.path.join("data", "raw")

FRANKFURTER_BASE = "https://api.frankfurter.dev/v1"  # versioned base
console = Console()

# ----------------------------
# Utilities
# ----------------------------
def read_cards_file(path: str) -> List[str]:
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

def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    # Per-vendor price history (native + ZAR)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            date            TEXT NOT NULL,      -- YYYY-MM-DD UTC
            card_id         INTEGER NOT NULL,
            card_name       TEXT NOT NULL,
            vendor          TEXT NOT NULL,      -- e.g., tcgplayer_price
            currency_native TEXT NOT NULL,      -- 'USD' or 'EUR'
            price_native    REAL,               -- nullable
            fx_rate_to_zar  REAL,               -- native -> ZAR
            fx_date         TEXT,               -- YYYY-MM-DD
            fx_source       TEXT,               -- 'Frankfurter(ECB)'
            price_zar       REAL,               -- price_native * fx_rate_to_zar
            PRIMARY KEY (date, card_id, vendor)
        )
    """)
    # Cached FX rates used
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fx_rates (
            fx_date TEXT NOT NULL,
            ccy     TEXT NOT NULL,   -- 'USD' or 'EUR'
            rate    REAL NOT NULL,   -- to ZAR
            source  TEXT NOT NULL,
            PRIMARY KEY (fx_date, ccy)
        )
    """)
    # Rarity-level averages from card_sets.set_price (USD → ZAR)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rarity_history (
            date               TEXT NOT NULL,      -- YYYY-MM-DD UTC
            card_id            INTEGER NOT NULL,
            card_name          TEXT NOT NULL,
            rarity             TEXT NOT NULL,      -- e.g., Ultra Rare
            sets_count         INTEGER NOT NULL,   -- contributing set entries
            avg_set_price_usd  REAL,
            fx_rate_to_zar     REAL,               -- USD -> ZAR
            fx_date            TEXT,
            fx_source          TEXT,
            avg_set_price_zar  REAL,
            PRIMARY KEY (date, card_id, rarity)
        )
    """)
    conn.commit()

def _request_with_retry(url: str, params: Dict[str, str] | None = None, max_retries: int = 3) -> dict:
    headers = {"User-Agent": "price-tracker/1.0 (+https://github.com/yourname/ygo-price-tracker)"}
    backoff = 1.6
    last = None
    for attempt in range(max_retries):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        last = r
        if r.status_code in (429, 502, 503, 504) and attempt < max_retries - 1:
            time.sleep(backoff**attempt)
            continue
        r.raise_for_status()
        return r.json()
    last.raise_for_status()

# ----------------------------
# YGOPRODeck fetch
# ----------------------------
def fetch_prices(card_names: List[str]) -> List[dict]:
    joined = "|".join(card_names)  # e.g., Baby Dragon|Time Wizard
    json_obj = _request_with_retry(API_URL, params={"name": joined})
    if "data" not in json_obj:
        raise RuntimeError(f"Unexpected API response: {json_obj}")
    return json_obj["data"]

# ----------------------------
# FX helpers (ECB via Frankfurter /v1)
# ----------------------------
def frankfurter_by_date(ccy: str, fx_date: str) -> dict:
    # GET /v1/YYYY-MM-DD?from=USD&to=ZAR
    url = f"{FRANKFURTER_BASE}/{fx_date}"
    return _request_with_retry(url, params={"from": ccy, "to": "ZAR"})

def fetch_daily_fx(fetch_date_utc: str) -> dict:
    """
    Use the previous working day's ECB rate; fall back up to 7 days.
    ECB publishes around ~16:00 CET on working days.  # docs
    """
    target = dt.datetime.strptime(fetch_date_utc, "%Y-%m-%d").date() - dt.timedelta(days=1)
    for _ in range(7):
        try:
            out = {}
            for ccy in ("USD", "EUR"):
                data = frankfurter_by_date(ccy, target.strftime("%Y-%m-%d"))
                out[ccy] = {
                    "rate": data["rates"]["ZAR"],
                    "fx_date": data["date"],
                    "source": "Frankfurter(ECB)"
                }
            return out
        except requests.HTTPError as e:
            if getattr(e.response, "status_code", None) == 404:
                target -= dt.timedelta(days=1)
                continue
            raise
    raise RuntimeError("No ECB FX available for the last 7 days.")

# ----------------------------
# Normalization (vendors + rarity) & persistence
# ----------------------------
def cache_raw(cards: List[dict], fetch_date: str) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{fetch_date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": fetch_date, "data": cards}, f, ensure_ascii=False, indent=2)

def normalize_vendor_rows(cards: List[dict], fetch_date: str, fx: dict) -> List[Tuple]:
    """Vendor prices → ZAR rows for price_history."""
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

def _rarity_aggregates_for_card(card: dict, usd_rate_info: dict) -> List[Tuple]:
    """
    Group a card's card_sets by set_rarity and compute average set_price (USD & ZAR).
    Returns tuples: (rarity, sets_count, avg_usd, rate, fx_date, fx_source, avg_zar)
    """
    sets = card.get("card_sets") or []
    by_rarity: Dict[str, List[float]] = {}
    for s in sets:
        rarity = s.get("set_rarity")
        price_usd_raw = s.get("set_price")  # USD per API guide
        if not rarity:
            continue
        try:
            price_usd = float(price_usd_raw) if price_usd_raw not in (None, "", "NaN") else None
        except ValueError:
            price_usd = None
        if price_usd is not None:
            by_rarity.setdefault(rarity, []).append(price_usd)

    out: List[Tuple] = []
    usd_rate = usd_rate_info.get("rate")
    fx_date = usd_rate_info.get("fx_date")
    fx_source = usd_rate_info.get("source")
    for rarity, prices in by_rarity.items():
        if not prices:
            continue
        avg_usd = sum(prices) / len(prices)
        avg_zar = avg_usd * usd_rate if (usd_rate is not None) else None
        out.append((rarity, len(prices), avg_usd, usd_rate, fx_date, fx_source, avg_zar))
    return out

def normalize_rarity_rows(cards: List[dict], fetch_date: str, fx: dict) -> List[Tuple]:
    """
    Build rows for rarity_history:
    (date, card_id, card_name, rarity, sets_count, avg_set_price_usd,
     fx_rate_to_zar, fx_date, fx_source, avg_set_price_zar)
    """
    rows: List[Tuple] = []
    usd_rate_info = fx.get("USD", {})
    for c in cards:
        card_id = c.get("id")
        card_name = c.get("name")
        for (rarity, sets_count, avg_usd, rate, fx_date, fx_source, avg_zar) in _rarity_aggregates_for_card(c, usd_rate_info):
            rows.append((
                fetch_date, card_id, card_name, rarity, sets_count, avg_usd,
                rate, fx_date, fx_source, avg_zar
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

def upsert_rarity(conn: sqlite3.Connection, rows: List[Tuple]) -> int:
    cur = conn.cursor()
    cur.executemany("""
        INSERT OR REPLACE INTO rarity_history
            (date, card_id, card_name, rarity, sets_count, avg_set_price_usd,
             fx_rate_to_zar, fx_date, fx_source, avg_set_price_zar)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    return cur.rowcount

# ----------------------------
# Latest snapshots & combine
# ----------------------------
def vendor_latest_wide(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Vendor ZAR prices wide (one row per card_id, card_name).
    Columns: card_id, card_name, amazon_price, cardmarket_price, coolstuffinc_price, ebay_price, tcgplayer_price
    """
    df = pd.read_sql_query("SELECT * FROM price_history", conn)
    if df.empty:
        return df
    latest = df["date"].max()
    df = df[df["date"] == latest]
    temp = df.assign(value=df["price_zar"])
    wide = (temp.pivot_table(index=["card_id", "card_name"],
                             columns="vendor",
                             values="value",
                             aggfunc="first")
                 .reset_index())
    # Ensure all expected vendor columns exist
    expected = ["amazon_price","cardmarket_price","coolstuffinc_price","ebay_price","tcgplayer_price"]
    for col in expected:
        if col not in wide.columns:
            wide[col] = pd.NA
    # Reorder vendor columns in requested order
    wide = wide[["card_id","card_name"] + expected]
    return wide.sort_values(["card_name","card_id"]).reset_index(drop=True)

def rarity_latest_long(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Rarity averages (latest) per (card_id, card_name, rarity).
    Columns: card_id, card_name, rarity, sets_count, avg_set_price_usd, avg_set_price_zar
    """
    df = pd.read_sql_query("SELECT * FROM rarity_history", conn)
    if df.empty:
        return df
    latest = df["date"].max()
    df = df[df["date"] == latest]
    keep = ["card_id","card_name","rarity","sets_count","avg_set_price_usd","avg_set_price_zar"]
    return df[keep].sort_values(["card_name","rarity"]).reset_index(drop=True)

def combined_latest(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Join rarity rows (one per card×rarity) to vendor wide (one per card).
    Result columns (exact order requested):
      card_id, card_name, rarity, sets_count,
      amazon_price, cardmarket_price, coolstuffinc_price, ebay_price, tcgplayer_price,
      avg_set_price_usd, avg_set_price_zar
    """
    v = vendor_latest_wide(conn)
    r = rarity_latest_long(conn)
    cols = ["card_id","card_name","rarity","sets_count",
            "amazon_price","cardmarket_price","coolstuffinc_price","ebay_price","tcgplayer_price",
            "avg_set_price_usd","avg_set_price_zar"]
    if v.empty and r.empty:
        return pd.DataFrame(columns=cols)
    if r.empty:
        # If no rarity data, still return rows (rarity/avg columns blank)
        v["rarity"] = pd.NA
        v["sets_count"] = pd.NA
        v["avg_set_price_usd"] = pd.NA
        v["avg_set_price_zar"] = pd.NA
        return v[cols]
    # Left-join rarity onto vendor prices
    merged = r.merge(v, on=["card_id","card_name"], how="left")
    # Ensure absent vendor columns appear (if a vendor lacked data that day)
    for c in ["amazon_price","cardmarket_price","coolstuffinc_price","ebay_price","tcgplayer_price"]:
        if c not in merged.columns:
            merged[c] = pd.NA
    return merged[cols].sort_values(["card_name","rarity"]).reset_index(drop=True)

# ----------------------------
# Pretty print helpers
# ----------------------------
def pretty_print(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        console.print(f"[red]No data to display for {title}.[/red]")
        return
    t = Table(title=title)
    for col in df.columns:
        t.add_column(col, overflow="fold")
    for _, row in df.iterrows():
        t.add_row(*[str(x if pd.notna(x) else "") for x in row.values])
    console.print(t)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="YGO-Price Tracker (ZAR + Rarity)")
    ap.add_argument("--cards", default="cards.txt")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--csv", default=LATEST_CSV_PATH)  # combined output path
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    # timezone-aware UTC date key
    fetch_date = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")

    card_list = read_cards_file(args.cards)
    cards = fetch_prices(card_list)
    if not args.no_cache:
        cache_raw(cards, fetch_date)

    conn = sqlite3.connect(args.db)
    ensure_schema(conn)

    fx = fetch_daily_fx(fetch_date)  # {'USD': {...}, 'EUR': {...}}
    upsert_fx(conn, fx)

    # Vendor rows (per vendor; ZAR)
    vendor_rows = normalize_vendor_rows(cards, fetch_date, fx)
    count_v = upsert_prices(conn, vendor_rows)

    # Rarity rows (avg USD + ZAR)
    rarity_rows = normalize_rarity_rows(cards, fetch_date, fx)
    count_r = upsert_rarity(conn, rarity_rows)

    # Build combined table and write the one CSV you asked for
    combined = combined_latest(conn)
    combined.to_csv(args.csv, index=False)

    # (Optional) also export the rarity snapshot table
    rarity_snapshot = rarity_latest_long(conn)
    rarity_snapshot.to_csv(RARITY_CSV_PATH, index=False)

    console.rule(f"[bold green]Upserted {count_v} vendor rows and {count_r} rarity rows for {fetch_date}[/bold green]")
    pretty_print(combined, "Latest (Combined): Vendors (ZAR) + Rarity Averages")
    console.print(f"\nWrote combined CSV to [bold]{args.csv}[/bold]")
    console.print(f"Wrote rarity snapshot CSV to [bold]{RARITY_CSV_PATH}[/bold]\n")

if __name__ == "__main__":
    main()
