#!/usr/bin/env python3
from __future__ import annotations
import os, sqlite3, argparse
from typing import List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# -----------------------
# Config used in app
# -----------------------
DB_PATH_DEFAULT = "prices.db"
VENDORS = ["cardmarket_price","tcgplayer_price","ebay_price","amazon_price","coolstuffinc_price"]

# -----------------------
# Data loading helpers
# -----------------------
@st.cache_data(show_spinner=False)
def load_price_history(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM price_history", conn, parse_dates=["date"])
    finally:
        conn.close()
    if df.empty:
        return df
    # Convert cents -> ZAR floats for display/plotting
    df["price_zar"] = df["price_zar_cents"].apply(lambda c: (c/100.0) if pd.notna(c) else None)
    return df

@st.cache_data(show_spinner=False)
def load_set_prices(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM set_prices", conn, parse_dates=["date"])
    finally:
        conn.close()
    return df

@st.cache_data(show_spinner=False)
def load_fx_usd(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        fx = pd.read_sql_query("SELECT fx_date, ccy, rate FROM fx_rates WHERE ccy='USD'", conn, parse_dates=["fx_date"])
    finally:
        conn.close()
    return fx.sort_values("fx_date")

def rarity_timeseries_zar(set_df: pd.DataFrame, fx_usd: pd.DataFrame, card_name: str) -> pd.DataFrame:
    """
    Build per-date rarity averages:
      group set_prices by (date, card, rarity) → mean USD
      merge_asof to get the USD->ZAR rate valid on/just before that date,
      compute ZAR = USD * rate.
    """
    if set_df.empty:
        return set_df

    sdf = set_df[set_df["card_name"] == card_name].copy()
    if sdf.empty:
        return sdf

    # mean USD per day×rarity (price_usd_cents may be NULL)
    sdf["usd"] = sdf["price_usd_cents"].apply(lambda c: (c/100.0) if pd.notna(c) else None)
    g = (sdf.groupby(["date","card_id","card_name","set_rarity"], dropna=False)["usd"]
            .mean().reset_index().rename(columns={"set_rarity":"rarity","usd":"avg_usd"}))

    if fx_usd.empty or g.empty:
        g["avg_zar"] = None
        return g

    # merge_asof to use the most recent fx_date <= run date
    fx = fx_usd.rename(columns={"fx_date":"fxd"}).sort_values("fxd")
    g = g.sort_values("date")
    g = pd.merge_asof(g, fx[["fxd","rate"]], left_on="date", right_on="fxd", direction="backward")
    g["avg_zar"] = g.apply(lambda r: (r["avg_usd"] * r["rate"]) if pd.notna(r["avg_usd"]) and pd.notna(r["rate"]) else None, axis=1)
    return g

# -----------------------
# Plot builder
# -----------------------
def make_figure(card_name: str,
                df: pd.DataFrame,
                show_vendors: List[str],
                rar_df: Optional[pd.DataFrame],
                rarities: List[str],
                date_min, date_max) -> go.Figure:
    fig = go.Figure()

    # Pretty labels for legend
    vendor_label = {
        "cardmarket_price": "Cardmarket",
        "tcgplayer_price":  "TCGplayer",
        "ebay_price":       "eBay",
        "amazon_price":     "Amazon",
        "coolstuffinc_price":"CoolStuffInc",
    }

    # ---- Vendor lines (ZAR)
    cdf = df[(df["card_name"] == card_name) & (df["vendor"].isin(show_vendors))].copy()
    if date_min is not None and date_max is not None:
        cdf = cdf[(cdf["date"] >= date_min) & (cdf["date"] <= date_max)]

    first_vendor_trace = True
    for vendor in show_vendors:
        sub = cdf[cdf["vendor"] == vendor].sort_values("date")
        if sub["price_zar"].notna().any():
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["price_zar"],
                mode="lines",
                name=vendor_label.get(vendor, vendor),
                legendgroup="vendors",
                legendgrouptitle_text="Vendors" if first_vendor_trace else None,
                hovertemplate="%{x|%Y-%m-%d}<br>ZAR %{y:.2f}<extra>"+vendor_label.get(vendor, vendor)+"</extra>",
                line=dict(width=2)
            ))
            first_vendor_trace = False

    # ---- Rarity overlays (optional)
    if rar_df is not None and not rar_df.empty and rarities:
        rdf = rar_df[(rar_df["card_name"] == card_name) & (rar_df["rarity"].isin(rarities))].copy()
        if date_min is not None and date_max is not None:
            rdf = rdf[(rdf["date"] >= date_min) & (rdf["date"] <= date_max)]

        first_rarity_trace = True
        for rar in rarities:
            sub = rdf[rdf["rarity"] == rar].sort_values("date")
            if sub["avg_zar"].notna().any():
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=sub["avg_zar"],
                    mode="lines",
                    name=f"avg {rar}",
                    legendgroup="rarity",
                    legendgrouptitle_text="Rarity avgs" if first_rarity_trace else None,
                    line=dict(width=2, dash="dash"),
                    hovertemplate="%{x|%Y-%m-%d}<br>Avg ZAR %{y:.2f}<extra>"+f"avg {rar}"+"</extra>",
                ))
                first_rarity_trace = False

    # ---- Layout polish
    fig.update_layout(
        template="plotly_white",                                     # theme/template :contentReference[oaicite:3]{index=3}
        title=dict(
            text=f"{card_name}",
            x=0.02, xanchor="left",
            pad=dict(t=8, b=6, l=4, r=4)
        ),
        hovermode="x unified",                                       # unified hover label :contentReference[oaicite:4]{index=4}
        legend=dict(orientation="h", x=0.0, xanchor="left",
                    y=-0.22, yanchor="top", bgcolor="rgba(0,0,0,0)"), # put legend below plot :contentReference[oaicite:5]{index=5}
        margin=dict(l=70, r=30, t=80, b=130),
        font=dict(size=14),
        title_font=dict(size=24),
        hoverlabel=dict(font_size=13)
    )

    # Axes: spacing + spikes
    fig.update_yaxes(title="Price (ZAR)", automargin=True, title_standoff=8, zeroline=False)
    fig.update_xaxes(
        title="Date (UTC)", automargin=True, title_standoff=8,
        # --- Monthly ticks by default:
        dtick="M1",                   # one tick per month
        ticklabelmode="period",       # center label in the month span  :contentReference[oaicite:6]{index=6}
        tickformat="%b\n%Y",          # e.g. "Mar" linebreak "2025"
        showspikes=True, spikemode="across", spikedash="dot",         # alignment aids :contentReference[oaicite:7]{index=7}
        rangeslider_visible=False
    )

    # Make sure the initial view matches the chosen sidebar range (whole range by default)
    if date_min is not None and date_max is not None:
        fig.update_xaxes(range=[pd.to_datetime(date_min), pd.to_datetime(date_max)])

    return fig


# -----------------------
# Streamlit UI
# -----------------------
def run_app(db_path: str):
    st.set_page_config(page_title="YGO Price Tracker", layout="wide")
    st.title("🪄 YGO Price Tracker — Interactive")

    ph = load_price_history(db_path)
    if ph.empty:
        st.warning("No data found. Run price_tracker.py first.")
        return

    sp = load_set_prices(db_path)
    fx = load_fx_usd(db_path)

    # Sidebar controls  :contentReference[oaicite:4]{index=4}
    with st.sidebar:
        st.header("Filters")
        cards = sorted(ph["card_name"].unique().tolist())
        card = st.selectbox("Card", cards)  # selectbox  :contentReference[oaicite:5]{index=5}

        vendors = st.multiselect("Vendors", VENDORS, default=VENDORS)

        # Rarity choices come from set_prices for this card
        rar_choices = sorted([r for r in sp.loc[sp["card_name"]==card, "set_rarity"].dropna().unique().tolist()])
        show_rarity = st.checkbox("Overlay rarity averages", value=True)
        selected_rar = st.multiselect("Rarities", rar_choices, default=rar_choices) if show_rarity else []

        # Date range
        dmin, dmax = ph["date"].min(), ph["date"].max()
        date_min, date_max = st.date_input("Date range", value=(dmin.date(), dmax.date()))
        # coerce to pandas Timestamps
        date_min = pd.to_datetime(date_min)
        date_max = pd.to_datetime(date_max)

    # Build rarity time series (ZAR) using merge_asof with USD fx
    rar_ts = rarity_timeseries_zar(sp, fx, card) if show_rarity else None

    # Main figure
    fig = make_figure(card, ph, vendors, rar_ts, selected_rar, date_min, date_max)

    # Draw chart  :contentReference[oaicite:6]{index=6}
    st.plotly_chart(fig, use_container_width=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH_DEFAULT, help="Path to prices.db")
    args, unknown = ap.parse_known_args()  # allow 'streamlit run plot_prices_app.py' extra args
    run_app(args.db)

if __name__ == "__main__":
    main()
