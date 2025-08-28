#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sqlite3
import matplotlib.pyplot as plt
import pandas as pd

# Vendors to draw (must match the names in price_history.vendor)
VENDORS = ["cardmarket_price","tcgplayer_price","ebay_price","amazon_price","coolstuffinc_price"]

def load_history(db_path: str) -> pd.DataFrame:
    """
    Load the vendor price history from SQLite and parse the 'date' column.
    Note: prices are stored as *cents* (INTEGER). We'll convert to ZAR floats for plotting.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM price_history", conn, parse_dates=["date"])
    finally:
        conn.close()
    if df.empty:
        return df
    # Convert cents -> ZAR (float) for plotting (safe: display only)
    df["price_zar"] = df["price_zar_cents"].apply(lambda c: (c / 100.0) if pd.notna(c) else None)
    return df

def sanitize(name: str) -> str:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "._- ")
    return safe.replace(" ", "_")

def plot_card(df: pd.DataFrame, outdir: str) -> str:
    """
    One PNG per card: X = date, Y = ZAR price, one line per vendor that has data.
    """
    card_name = df["card_name"].iloc[0]
    fig, ax = plt.subplots()  # one chart per figure (no subplots)
    for vendor in VENDORS:
        sub = df[df["vendor"] == vendor].sort_values("date")
        if sub["price_zar"].notna().any():
            # Let Matplotlib pick default colors, and it will format datetimes on X.  :contentReference[oaicite:1]{index=1}
            ax.plot(sub["date"], sub["price_zar"], label=vendor)
    ax.set_title(f"{card_name} — Price History (ZAR)")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Price (ZAR)")
    ax.legend()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{sanitize(card_name)}.png")
    # Tight bbox reduces extra margins; dpi gives a crisp image.  :contentReference[oaicite:2]{index=2}
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prices.db", help="Path to SQLite DB")
    ap.add_argument("--out", default="charts", help="Output directory for PNGs")
    ap.add_argument("--filter", nargs="*", help="Optional card name filters (exact match)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_history(args.db)
    if df.empty:
        print("No data found. Run price_tracker.py first.")
        return

    if args.filter:
        df = df[df["card_name"].isin(args.filter)]
        if df.empty:
            print("No matching cards for provided --filter.")
            return

    # Group by card and plot each
    for (_, _), group in df.groupby(["card_id", "card_name"]):
        print("Plotting:", group["card_name"].iloc[0])
        plot_card(group, args.out)

if __name__ == "__main__":
    main()
