#!/usr/bin/env python3
"""
Yu-Gi-Oh! Card Price Tracker (ZAR) — normalized, cents-accurate, with auto-migration
(now with rotating logs)

- Ingests from YGOPRODeck v7 cardinfo (batched exact names via |)
- Stores vendor prices (EUR/USD -> ZAR) as integer cents (no float drift)
- Stores per-print set prices (USD cents) in a normalized table (one row per set_code)
- Fetches ECB FX via Frankfurter /v1 (latest WORKING day ~16:00 CET) with fallback
- Auto-migrates old DB schema (REAL columns) to cents schema if needed
- SQLite tuned (WAL, foreign_keys ON, indexes)
- Exports ONE combined CSV (prices_latest.csv)
- Logs to logs/run.log with size-based rotation
"""

from __future__ import annotations
import argparse
import datetime as dt
from datetime import timezone
import json
import os
import sqlite3
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from rich.console import Console
from rich.table import Table

# --------------------------------
# Constants & configuration
# --------------------------------
API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"   # v7; card_prices + card_sets
FRANKFURTER_BASE = "https://api.frankfurter.dev/v1"         # latest working day ~16:00 CET

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

DB_PATH = "prices.db"
LATEST_CSV_PATH = "prices_latest.csv"
RAW_DIR = os.path.join("data", "raw")

# logging config
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "run.log")
LOG_MAX_BYTES = 512_000   # ~0.5 MB per file
LOG_BACKUPS = 3

console = Console()
logger = logging.getLogger("ygo_tracker")

def _setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger.setLevel(logging.INFO)

    # Rotating file handler (size-based rotation)
    fh = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
    )
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Echo to console too
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(ch)

    logger.info("Logging initialized → %s (maxBytes=%s, backups=%s)", LOG_FILE, LOG_MAX_BYTES, LOG_BACKUPS)

# --------------------------------
# Money helpers (store cents)
# --------------------------------
TWOPLACES = Decimal("0.01")

def parse_amount_2dp(value: Optional[str | float]) -> Optional[Decimal]:
    if value in (None, "", "NaN"):
        return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return d.quantize(TWOPLACES, rounding=ROUND_HALF_UP)

def to_cents(value_2dp: Optional[Decimal]) -> Optional[int]:
    if value_2dp is None:
        return None
    return int((value_2dp * 100).to_integral_value(rounding=ROUND_HALF_UP))

def from_cents(cents: Optional[int]) -> Optional[Decimal]:
    if cents is None:
        return None
    return (Decimal(cents) / Decimal(100)).quantize(TWOPLACES)

# --------------------------------
# SQLite helpers
# --------------------------------
def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]  # column names

def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    # Vendor price history (store cents)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            date                TEXT NOT NULL,   -- YYYY-MM-DD UTC (run date)
            card_id             INTEGER NOT NULL,
            card_name           TEXT NOT NULL,
            vendor              TEXT NOT NULL,   -- tcgplayer_price / etc
            currency_native     TEXT NOT NULL,   -- 'USD' / 'EUR'
            price_native_cents  INTEGER,         -- nullable
            fx_rate_to_zar      REAL,            -- multiplier (native -> ZAR)
            fx_date             TEXT,            -- ECB ref date (YYYY-MM-DD)
            fx_source           TEXT,            -- 'Frankfurter(ECB)'
            price_zar_cents     INTEGER,         -- price_native * rate (cents)
            PRIMARY KEY (date, card_id, vendor)
        )
    """)
    # Per-print set prices (normalized, USD cents)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS set_prices (
            date            TEXT NOT NULL,
            card_id         INTEGER NOT NULL,
            card_name       TEXT NOT NULL,
            set_code        TEXT NOT NULL,
            set_name        TEXT,
            set_rarity      TEXT,
            price_usd_cents INTEGER,
            PRIMARY KEY (date, card_id, set_code)
        )
    """)
    # Cached FX used in a run
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

def tune_sqlite(conn: sqlite3.Connection) -> None:
    """Enable FK, set WAL (persistent), add helpful indexes."""
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_price_date_card ON price_history(date, card_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_price_card_vendor ON price_history(card_id, vendor)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_set_card_rarity ON set_prices(card_id, set_rarity)")
    conn.commit()

def migrate_price_history_if_needed(conn: sqlite3.Connection) -> None:
    """
    If price_history exists but lacks cents columns, migrate:
    - Create price_history_new (cents schema)
    - Copy & convert from old columns (price_native -> price_native_cents, price_zar/price -> price_zar_cents)
    - Swap tables (rename)
    """
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_history'")
    exists = cur.fetchone() is not None
    if not exists:
        return  # nothing to migrate

    cols = table_columns(conn, "price_history")
    if "price_native_cents" in cols and "price_zar_cents" in cols:
        return  # already migrated

    console.print("[yellow]Migrating price_history to cents-based schema...[/yellow]")
    logger.info("Migrating price_history to cents-based schema")

    # Create new table with desired schema
    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_history_new (
            date                TEXT NOT NULL,
            card_id             INTEGER NOT NULL,
            card_name           TEXT NOT NULL,
            vendor              TEXT NOT NULL,
            currency_native     TEXT NOT NULL,
            price_native_cents  INTEGER,
            fx_rate_to_zar      REAL,
            fx_date             TEXT,
            fx_source           TEXT,
            price_zar_cents     INTEGER,
            PRIMARY KEY (date, card_id, vendor)
        )
    """)

    # Figure out which old numeric columns we have
    has_price_native = "price_native" in cols
    has_price_zar = "price_zar" in cols
    has_price_plain = "price" in cols  # very early versions

    def cents_expr(colname: str) -> str:
        # ROUND(x*100) -> INTEGER cents; NULL should stay NULL
        return f"CASE WHEN {colname} IS NULL THEN NULL ELSE CAST(ROUND({colname} * 100) AS INTEGER) END"

    native_expr = cents_expr("price_native") if has_price_native else "NULL"
    if has_price_zar:
        zar_expr = cents_expr("price_zar")
    elif has_price_plain:
        zar_expr = cents_expr("price")
    else:
        zar_expr = "NULL"

    # Copy data
    cur.execute(f"""
        INSERT OR REPLACE INTO price_history_new
            (date, card_id, card_name, vendor, currency_native,
             price_native_cents, fx_rate_to_zar, fx_date, fx_source, price_zar_cents)
        SELECT
            date, card_id, card_name, vendor, currency_native,
            {native_expr} AS price_native_cents,
            fx_rate_to_zar, fx_date, fx_source,
            {zar_expr} AS price_zar_cents
        FROM price_history
    """)

    # Swap
    cur.execute("ALTER TABLE price_history RENAME TO price_history_old")
    cur.execute("ALTER TABLE price_history_new RENAME TO price_history")
    cur.execute("DROP TABLE price_history_old")
    conn.commit()

    # Recreate indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_price_date_card ON price_history(date, card_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_price_card_vendor ON price_history(card_id, vendor)")
    conn.commit()
    console.print("[green]Migration complete.[/green]")
    logger.info("Migration complete")

# --------------------------------
# HTTP (with retry)
# --------------------------------
def _request_with_retry(url: str, params: Dict[str, str] | None = None, max_retries: int = 3) -> dict:
    headers = {"User-Agent": "ygo-price-tracker/2.1 (+https://github.com/yourname/ygo-price-tracker)"}
    backoff = 1.6
    last = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            last = r
            if r.status_code in (429, 502, 503, 504) and attempt < max_retries - 1:
                delay = backoff ** attempt
                logger.warning("HTTP %s from %s (attempt %s/%s). Sleeping %.2fs and retrying…",
                               r.status_code, url, attempt+1, max_retries, delay)
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = backoff ** attempt
                logger.warning("Request error on %s (attempt %s/%s): %s — retrying in %.2fs",
                               url, attempt+1, max_retries, e, delay)
                time.sleep(delay)
                continue
            logger.exception("Request failed after %s attempts: %s", max_retries, url)
            raise
    # Shouldn't reach here; keep for safety
    if last is not None:
        last.raise_for_status()
    raise RuntimeError("Unexpected HTTP retry flow")

# --------------------------------
# Fetchers
# --------------------------------
def fetch_prices(card_names: List[str]) -> List[dict]:
    joined = "|".join(card_names)  # pipe-separated exact names
    logger.info("Fetching prices for %d cards from YGOPRODeck", len(card_names))
    json_obj = _request_with_retry(API_URL, params={"name": joined})
    if "data" not in json_obj:
        logger.error("Unexpected API response (no 'data'): %s", json_obj)
        raise RuntimeError(f"Unexpected API response: {json_obj}")
    return json_obj["data"]

def frankfurter_by_date(ccy: str, fx_date: str) -> dict:
    url = f"{FRANKFURTER_BASE}/{fx_date}"
    return _request_with_retry(url, params={"from": ccy, "to": "ZAR"})

def fetch_daily_fx(fetch_date_utc: str) -> dict:
    """
    Use prior working day ECB rate; fall back up to 7 days (weekends/holidays).
    Frankfurter returns 'latest working day' around ~16:00 CET.
    """
    target = dt.datetime.strptime(fetch_date_utc, "%Y-%m-%d").date() - dt.timedelta(days=1)
    logger.info("Fetching FX rates (USD, EUR) for run date=%s using <=7-day fallback", fetch_date_utc)
    for _ in range(7):
        try:
            out = {}
            for ccy in ("USD", "EUR"):
                data = frankfurter_by_date(ccy, target.strftime("%Y-%m-%d"))
                out[ccy] = {"rate": data["rates"]["ZAR"], "fx_date": data["date"], "source": "Frankfurter(ECB)"}
            logger.info("FX resolved: USD=%s (date %s), EUR=%s (date %s)",
                        out["USD"]["rate"], out["USD"]["fx_date"], out["EUR"]["rate"], out["EUR"]["fx_date"])
            return out
        except requests.HTTPError as e:
            if getattr(e.response, "status_code", None) == 404:
                logger.warning("No FX for %s (404). Trying previous day…", target)
                target -= dt.timedelta(days=1)
                continue
            logger.exception("HTTP error while fetching FX")
            raise
    logger.error("No ECB FX available for the last 7 days.")
    raise RuntimeError("No ECB FX available for the last 7 days.")

# --------------------------------
# Normalizers (to DB rows)
# --------------------------------
def cache_raw(cards: List[dict], fetch_date: str) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{fetch_date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": fetch_date, "data": cards}, f, ensure_ascii=False, indent=2)
    logger.info("Cached raw API JSON → %s", path)

def normalize_vendor_rows(cards: List[dict], fetch_date: str, fx: dict) -> List[Tuple]:
    rows: List[Tuple] = []
    for c in cards:
        cid = c.get("id")
        cname = c.get("name")
        prices = (c.get("card_prices") or [{}])[0]
        for vendor in VENDORS:
            amt = parse_amount_2dp(prices.get(vendor))
            native_cents = to_cents(amt)
            ccy = VENDOR_CCY[vendor]
            rate_info = fx.get(ccy, {})
            rate = rate_info.get("rate")
            if native_cents is not None and rate is not None:
                price_zar = (Decimal(native_cents) / Decimal(100)) * Decimal(str(rate))
                price_zar_cents = to_cents(price_zar.quantize(TWOPLACES, rounding=ROUND_HALF_UP))
            else:
                price_zar_cents = None
            rows.append((
                fetch_date, cid, cname, vendor,
                ccy, native_cents,
                rate, rate_info.get("fx_date"), rate_info.get("source"),
                price_zar_cents
            ))
    return rows

def normalize_set_price_rows(cards: List[dict], fetch_date: str) -> List[Tuple]:
    rows: List[Tuple] = []
    for c in cards:
        cid = c.get("id")
        cname = c.get("name")
        for s in (c.get("card_sets") or []):
            scode = s.get("set_code")
            sname = s.get("set_name")
            rarity = s.get("set_rarity")
            usd = parse_amount_2dp(s.get("set_price"))
            usd_cents = to_cents(usd)
            rows.append((fetch_date, cid, cname, scode, sname, rarity, usd_cents))
    return rows

# --------------------------------
# Upserts
# --------------------------------
def upsert_fx(conn: sqlite3.Connection, fx: dict) -> None:
    cur = conn.cursor()
    for ccy, info in fx.items():
        cur.execute("""
            INSERT OR REPLACE INTO fx_rates (fx_date, ccy, rate, source)
            VALUES (?, ?, ?, ?)
        """, (info["fx_date"], ccy, info["rate"], info["source"]))
    conn.commit()
    logger.info("Upserted FX rows: %d", len(fx))

def upsert_vendor_prices(conn: sqlite3.Connection, rows: List[Tuple]) -> int:
    cur = conn.cursor()
    cur.executemany("""
        INSERT OR REPLACE INTO price_history
            (date, card_id, card_name, vendor, currency_native, price_native_cents,
             fx_rate_to_zar, fx_date, fx_source, price_zar_cents)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    logger.info("Upserted vendor price rows: %d", cur.rowcount)
    return cur.rowcount

def upsert_set_prices(conn: sqlite3.Connection, rows: List[Tuple]) -> int:
    cur = conn.cursor()
    cur.executemany("""
        INSERT OR REPLACE INTO set_prices
            (date, card_id, card_name, set_code, set_name, set_rarity, price_usd_cents)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    logger.info("Upserted set price rows: %d", cur.rowcount)
    return cur.rowcount

# --------------------------------
# Snapshots & combine
# --------------------------------
def vendor_latest_wide(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Latest vendor prices (ZAR) wide by card.
    Returns: card_id, card_name, amazon_price, cardmarket_price, coolstuffinc_price, ebay_price, tcgplayer_price
    """
    df = pd.read_sql_query("SELECT * FROM price_history", conn)
    if df.empty:
        return df
    latest = df["date"].max()
    df = df[df["date"] == latest].copy()
    df["value"] = df["price_zar_cents"].apply(lambda c: float(from_cents(c)) if c is not None else None)
    wide = (df.pivot_table(index=["card_id", "card_name"], columns="vendor", values="value", aggfunc="first")
              .reset_index())
    expected = ["amazon_price", "cardmarket_price", "coolstuffinc_price", "ebay_price", "tcgplayer_price"]
    for col in expected:
        if col not in wide.columns:
            wide[col] = pd.NA
    return wide[["card_id", "card_name"] + expected].sort_values(["card_name", "card_id"]).reset_index(drop=True)

def rarity_latest_from_sets(conn: sqlite3.Connection, usd_rate: Optional[float]) -> pd.DataFrame:
    """
    Aggregate per (card, rarity) from normalized set_prices (latest date).
    Returns: card_id, card_name, rarity, sets_count, avg_set_price_usd, avg_set_price_zar
    """
    df = pd.read_sql_query("SELECT * FROM set_prices", conn)
    if df.empty:
        return df
    latest = df["date"].max()
    df = df[df["date"] == latest].copy()
    # cents -> Decimal USD for each row
    df["usd_amount"] = df["price_usd_cents"].apply(lambda c: from_cents(c))
    agg = (df.groupby(["card_id", "card_name", "set_rarity"], dropna=False)["usd_amount"]
             .agg(["count", "mean"]).reset_index())
    agg.rename(columns={"set_rarity": "rarity", "count": "sets_count", "mean": "avg_set_price_usd"}, inplace=True)
    # Compute ZAR avg using USD->ZAR rate for the run
    if usd_rate is not None:
        agg["avg_set_price_zar"] = agg["avg_set_price_usd"].apply(
            lambda x: float((Decimal(str(x)) * Decimal(str(usd_rate))).quantize(TWOPLACES, rounding=ROUND_HALF_UP)) if pd.notna(x) else None
        )
    else:
        agg["avg_set_price_zar"] = pd.NA
    # Convert USD Decimal to float for CSV
    agg["avg_set_price_usd"] = agg["avg_set_price_usd"].apply(
        lambda x: float(Decimal(str(x)).quantize(TWOPLACES)) if pd.notna(x) else None
    )
    return agg.sort_values(["card_name", "rarity"]).reset_index(drop=True)

def combined_latest(conn: sqlite3.Connection, fx: dict) -> pd.DataFrame:
    """
    Join rarity rows (card×rarity) to vendor wide (card).
    Final columns (exact order):
      card_id, card_name, rarity, sets_count,
      amazon_price, cardmarket_price, coolstuffinc_price, ebay_price, tcgplayer_price,
      avg_set_price_usd, avg_set_price_zar
    """
    v = vendor_latest_wide(conn)
    usd_rate = fx.get("USD", {}).get("rate")
    r = rarity_latest_from_sets(conn, usd_rate)
    cols = ["card_id", "card_name", "rarity", "sets_count",
            "amazon_price", "cardmarket_price", "coolstuffinc_price", "ebay_price", "tcgplayer_price",
            "avg_set_price_usd", "avg_set_price_zar"]
    if v.empty and r.empty:
        return pd.DataFrame(columns=cols)
    if r.empty:
        # still return vendor rows with blank rarity/avg columns
        v["rarity"] = pd.NA
        v["sets_count"] = pd.NA
        v["avg_set_price_usd"] = pd.NA
        v["avg_set_price_zar"] = pd.NA
        return v[cols]
    merged = r.merge(v, on=["card_id", "card_name"], how="left")
    for c in ["amazon_price", "cardmarket_price", "coolstuffinc_price", "ebay_price", "tcgplayer_price"]:
        if c not in merged.columns:
            merged[c] = pd.NA
    return merged[cols].sort_values(["card_name", "rarity"]).reset_index(drop=True)

# --------------------------------
# Pretty print
# --------------------------------
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

def read_cards_file(path: str) -> list[str]:
    """
    Reads one exact card name per line from cards.txt.
    Ignores blank lines and lines starting with '#'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Card list file not found: {path}")
    names: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                names.append(s)
    if not names:
        raise ValueError(f"No card names found in {path}")
    return names

# --------------------------------
# Main
# --------------------------------
def main():
    ap = argparse.ArgumentParser(description="YGO Price Tracker — ZAR + rarity, normalized, cents-accurate")
    ap.add_argument("--cards", default="cards.txt", help="Path to cards.txt (one exact name per line)")
    ap.add_argument("--db", default=DB_PATH, help="SQLite DB path")
    ap.add_argument("--csv", default=LATEST_CSV_PATH, help="Output combined CSV")
    ap.add_argument("--no-cache", action="store_true", help="Skip writing raw API JSON to data/raw")
    args = ap.parse_args()

    _setup_logging()
    logger.info("Run started")

    # UTC date key (works on Python 3.10+)
    fetch_date = dt.datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info("Fetch date (UTC): %s", fetch_date)

    try:
        card_names = read_cards_file(args.cards)
        logger.info("Loaded %d card names from %s", len(card_names), args.cards)

        cards = fetch_prices(card_names)
        if not args.no_cache:
            cache_raw(cards, fetch_date)

        conn = sqlite3.connect(args.db)
        ensure_schema(conn)
        migrate_price_history_if_needed(conn)  # <-- auto-migrate old schema if present
        tune_sqlite(conn)

        fx = fetch_daily_fx(fetch_date)     # {'USD': {rate, fx_date, source}, 'EUR': {...}}
        upsert_fx(conn, fx)

        vendor_rows = normalize_vendor_rows(cards, fetch_date, fx)
        set_rows = normalize_set_price_rows(cards, fetch_date)
        inserted_v = upsert_vendor_prices(conn, vendor_rows)
        inserted_s = upsert_set_prices(conn, set_rows)

        combined = combined_latest(conn, fx)
        combined.to_csv(args.csv, index=False)
        logger.info("Wrote combined CSV → %s (%d rows)", args.csv, len(combined))

        console.rule(f"[bold green]Upserted {inserted_v} vendor rows and {inserted_s} set rows for {fetch_date}[/bold green]")
        pretty_print(combined, "Latest (Combined): Vendors (ZAR) + Rarity Averages")

        console.print(f"\nWrote combined CSV to [bold]{args.csv}[/bold]")
        console.print("DB tuned with WAL + indexes; money stored as cents for accuracy.\n")
        logger.info("Run completed successfully")

    except Exception as e:
        logger.exception("Run failed: %s", e)
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
