#!/usr/bin/env python3
# ── YGO Price Tracker · Streamlit UI (polished) ────────────────────────────────
from __future__ import annotations
import os, sqlite3, argparse
from typing import List, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="YGO Price Tracker",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded",
)  # docs: st.set_page_config must be first call

# ── App constants (kept from your app) ────────────────────────────────────────
DB_PATH_DEFAULT = "prices.db"
VENDORS = ["cardmarket_price","tcgplayer_price","ebay_price","amazon_price","coolstuffinc_price"]

# ── Data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_price_history(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM price_history", conn, parse_dates=["date"])
    finally:
        conn.close()
    if df.empty:
        return df
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
        fx = pd.read_sql_query(
            "SELECT fx_date, ccy, rate FROM fx_rates WHERE ccy='USD'",
            conn, parse_dates=["fx_date"]
        )
    finally:
        conn.close()
    return fx.sort_values("fx_date")

# ── Helper: rarity time series (ZAR) ──────────────────────────────────────────
def rarity_timeseries_zar(set_df: pd.DataFrame, fx_usd: pd.DataFrame, card_name: str) -> pd.DataFrame:
    if set_df.empty:
        return set_df
    sdf = set_df[set_df["card_name"] == card_name].copy()
    if sdf.empty:
        return sdf
    sdf["usd"] = sdf["price_usd_cents"].apply(lambda c: (c/100.0) if pd.notna(c) else None)
    g = (sdf.groupby(["date","card_id","card_name","set_rarity"], dropna=False)["usd"]
            .mean().reset_index().rename(columns={"set_rarity":"rarity","usd":"avg_usd"}))
    if fx_usd.empty or g.empty:
        g["avg_zar"] = None
        return g
    fx = fx_usd.rename(columns={"fx_date":"fxd"}).sort_values("fxd")
    g = g.sort_values("date")
    g = pd.merge_asof(g, fx[["fxd","rate"]], left_on="date", right_on="fxd", direction="backward")
    g["avg_zar"] = g.apply(
        lambda r: (r["avg_usd"] * r["rate"]) if pd.notna(r["avg_usd"]) and pd.notna(r["rate"]) else None, axis=1
    )
    return g

# ── Plot builder ──────────────────────────────────────────────────────────────
def make_figure(
    card_name: str,
    df: pd.DataFrame,
    show_vendors: List[str],
    rar_df: Optional[pd.DataFrame],
    rarities: List[str],
    date_min, date_max,
    use_rangeslider: bool = False,
) -> go.Figure:
    fig = go.Figure()

    vendor_label = {
        "cardmarket_price": "Cardmarket",
        "tcgplayer_price":  "TCGplayer",
        "ebay_price":       "eBay",
        "amazon_price":     "Amazon",
        "coolstuffinc_price":"CoolStuffInc",
    }

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

    # Layout polish: unified hover, monthly ticks, spacious legend below
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",  # unified tooltip along vertical cursor
        title=dict(text=f"{card_name}", x=0.01, xanchor="left"),
        legend=dict(orientation="h", x=0.0, xanchor="left", y=-0.22, yanchor="top", bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=70, r=30, t=60, b=120),
        font=dict(size=14),
        title_font=dict(size=22),
        hoverlabel=dict(font_size=13),
    )
    fig.update_yaxes(title="Price (ZAR)", automargin=True, title_standoff=8, zeroline=False)
    fig.update_xaxes(
        title="Date (UTC)", automargin=True, title_standoff=8,
        dtick="M1", ticklabelmode="period", tickformat="%b\n%Y",
        showspikes=True, spikemode="across", spikedash="dot",
        rangeslider_visible=use_rangeslider,
    )
    if date_min is not None and date_max is not None:
        fig.update_xaxes(range=[pd.to_datetime(date_min), pd.to_datetime(date_max)])

    return fig

# ── App runner ────────────────────────────────────────────────────────────────
def run_app(db_path: str):
    st.title("🪄 YGO Price Tracker")
    st.caption("Interactive vendor prices in ZAR with optional rarity overlays.")

    ph = load_price_history(db_path)
    if ph.empty:
        st.warning("No data found. Run `price_tracker.py` to populate the database.")
        return

    sp = load_set_prices(db_path)
    fx = load_fx_usd(db_path)

    # Sidebar controls
    with st.sidebar:
        st.header("Filters")
        cards = sorted(ph["card_name"].unique().tolist())
        card = st.selectbox("Card", cards, help="Pick a card to visualize.")

        vendors = st.multiselect("Vendors", VENDORS, default=VENDORS, help="Show/hide vendor lines.")

        rar_choices = sorted([r for r in sp.loc[sp["card_name"]==card, "set_rarity"].dropna().unique().tolist()])
        show_rarity = st.checkbox("Overlay rarity averages", value=True)
        selected_rar = st.multiselect("Rarities", rar_choices, default=rar_choices) if show_rarity else []

        dmin, dmax = ph["date"].min(), ph["date"].max()
        date_min, date_max = st.date_input("Date range", value=(dmin.date(), dmax.date()))
        date_min = pd.to_datetime(date_min); date_max = pd.to_datetime(date_max)

        use_rangeslider = st.toggle("Show range slider", value=False)

    # KPI row
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Last data date", ph["date"].max().strftime("%Y-%m-%d"))
    with c2: st.metric("Cards tracked", f"{ph['card_id'].nunique():,}")
    with c3: st.metric("Vendors visible", f"{len(vendors)} / {len(VENDORS)}")

    # Precompute rarity series (if needed)
    rar_ts = rarity_timeseries_zar(sp, fx, card) if show_rarity else None

    # Tabs: Overview (chart), Data (table), About
    tab_overview, tab_data, tab_about = st.tabs(["📈 Overview", "🧮 Data", "ℹ️ About"])

    with tab_overview:
        fig = make_figure(card, ph, vendors, rar_ts, selected_rar, date_min, date_max, use_rangeslider)
        st.plotly_chart(fig, use_container_width=True)

    with tab_data:
        # Filtered vendor rows shown in chart (for quick CSV export)
        filt = ph[(ph["card_name"]==card) & (ph["vendor"].isin(vendors))]
        filt = filt[(filt["date"]>=date_min) & (filt["date"]<=date_max)].sort_values(["vendor","date"])
        st.dataframe(
            filt[["date","card_name","vendor","price_zar"]].rename(columns={"price_zar": "price_zar_float"}),
            use_container_width=True, height=380
        )
        csv = filt.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", csv, file_name=f"{card}_filtered.csv", mime="text/csv")

    with tab_about:
        st.markdown(
            """
            **How to use:** Pick a card, choose vendors, optionally overlay rarity averages, and adjust the date range.
            The chart uses unified hover and monthly tick labels for readability.  
            **Tip:** Enable the range slider for quick zooming.
            """
        )
        st.caption("Streamlit widgets (selectbox, multiselect, date range) and caching power this UI; Plotly handles interactivity.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH_DEFAULT, help="Path to prices.db")
    args, _ = ap.parse_known_args()   # allow: streamlit run plot_prices_app.py
    run_app(args.db)

if __name__ == "__main__":
    main()
