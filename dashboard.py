"""
dashboard.py
============
Technical Signal Streamlit Dashboard

Front-end for data_ingestion.py + indicators_signals.py.
Fetches FMP OHLCV data, computes 16 trading signals, and presents
interactive charts across four tabs.

Run:
    streamlit run dashboard.py
"""

# ---------------------------------------------------------------------------
# Block 1 — Imports & page config
# ---------------------------------------------------------------------------

import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_ingestion import FMPAuthError
from indicators_signals import build_signals_and_returns, SIGNAL_NAMES

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Technical Signal Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Block 2 — Cached analysis wrapper
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _run_analysis(symbol, start_str, end_str, interval, api_key):
    """Run full pipeline; cache results for 1 hour per unique parameter set."""
    return build_signals_and_returns(symbol, start_str, end_str, interval, api_key or None)


# ---------------------------------------------------------------------------
# Block 3 — Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Technical Signal Dashboard")
    st.markdown("---")

    api_key = st.text_input(
        "FMP API Key",
        type="password",
        value=os.environ.get("FMP_API_KEY", ""),
        help="Get a free key at financialmodelingprep.com",
    )
    ticker = st.text_input("Ticker", value="AAPL")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.Timestamp("2023-01-01"))
    with col2:
        end_date = st.date_input("End", value=pd.Timestamp("2025-01-01"))

    interval = st.selectbox(
        "Interval",
        ["1day", "1hour", "4hour", "30min", "15min", "5min", "1min"],
    )

    run_button = st.button("Run Analysis", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Block 4 — Run trigger → session state
# ---------------------------------------------------------------------------

if run_button:
    if not ticker.strip():
        st.sidebar.error("Ticker cannot be empty.")
    elif start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
    else:
        with st.spinner(f"Fetching {ticker.strip().upper()} data and computing signals…"):
            try:
                df, summary_df = _run_analysis(
                    ticker.strip().upper(),
                    str(start_date),
                    str(end_date),
                    interval,
                    api_key.strip(),
                )
                if df.empty:
                    st.warning("No data returned for the requested parameters.")
                else:
                    st.session_state["df"] = df
                    st.session_state["summary_df"] = summary_df
                    st.session_state["ticker"] = ticker.strip().upper()
                    st.session_state["interval"] = interval
            except FMPAuthError as exc:
                st.error(f"Authentication error: {exc}")
            except Exception as exc:
                st.error(f"Error: {exc}")

if "df" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **Run Analysis** to get started.")
    st.stop()

# Unpack session state
df: pd.DataFrame = st.session_state["df"]
summary_df: pd.DataFrame = st.session_state["summary_df"]
ticker: str = st.session_state["ticker"]
interval: str = st.session_state["interval"]

st.title(f"{ticker} — Technical Signal Dashboard")
st.caption(f"Interval: {interval}  |  {df.index[0].date()} → {df.index[-1].date()}  |  {len(df):,} bars")


# ---------------------------------------------------------------------------
# Block 5 — Four tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Price & Overlays", "📊 Oscillators", "🏆 Signal Performance", "📉 Equity Curves"]
)


# ---------------------------------------------------------------------------
# Tab 1: Price & Overlays
# ---------------------------------------------------------------------------

with tab1:
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="green",
            decreasing_line_color="red",
        ),
        row=1,
        col=1,
    )

    # SMA 20
    if "sma_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["sma_20"], name="SMA 20",
                line=dict(color="blue", width=1.5),
            ),
            row=1, col=1,
        )

    # EMA 20
    if "ema_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["ema_20"], name="EMA 20",
                line=dict(color="orange", width=1.5),
            ),
            row=1, col=1,
        )

    # Bollinger Bands
    if all(c in df.columns for c in ["bb_upper", "bb_mid", "bb_lower"]):
        for col_name, label in [("bb_upper", "BB Upper"), ("bb_mid", "BB Mid"), ("bb_lower", "BB Lower")]:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[col_name], name=label,
                    line=dict(color="gray", width=1, dash="dash"),
                ),
                row=1, col=1,
            )

    # VWAP
    if "vwap_daily_cum" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["vwap_daily_cum"], name="VWAP",
                line=dict(color="purple", width=1.5),
            ),
            row=1, col=1,
        )

    # Volume bars (green = up candle, red = down candle)
    bar_colors = [
        "green" if c >= o else "red"
        for c, o in zip(df["close"].values, df["open"].values)
    ]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["volume"], name="Volume",
            marker_color=bar_colors, showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=700,
        title=f"{ticker} — Price & Overlays ({interval})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Disable rangeslider on the volume subplot's x-axis too
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Oscillators
# ---------------------------------------------------------------------------

with tab2:
    fig2 = make_subplots(
        rows=5,
        cols=1,
        row_heights=[0.22, 0.18, 0.18, 0.22, 0.20],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["RSI", "Stochastic", "MFI", "MACD", "Aroon"],
    )

    # ── Row 1: RSI ──────────────────────────────────────────────────────────
    if "rsi_14" in df.columns:
        fig2.add_trace(
            go.Scatter(x=df.index, y=df["rsi_14"], name="RSI 14",
                       line=dict(color="blue", width=1.5)),
            row=1, col=1,
        )
    if "rsi_avg_14" in df.columns:
        fig2.add_trace(
            go.Scatter(x=df.index, y=df["rsi_avg_14"], name="RSI Avg 14",
                       line=dict(color="orange", width=1.5)),
            row=1, col=1,
        )
    if all(c in df.columns for c in ["rsi_bb_upper", "rsi_bb_mid", "rsi_bb_lower"]):
        for col_name, label in [
            ("rsi_bb_upper", "RSI BB Upper"),
            ("rsi_bb_mid", "RSI BB Mid"),
            ("rsi_bb_lower", "RSI BB Lower"),
        ]:
            fig2.add_trace(
                go.Scatter(x=df.index, y=df[col_name], name=label,
                           line=dict(color="gray", width=1, dash="dot")),
                row=1, col=1,
            )
    # Oversold / overbought zones
    fig2.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.07, line_width=0, row=1, col=1)
    fig2.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.07, line_width=0, row=1, col=1)
    fig2.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=1, col=1)
    fig2.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=1, col=1)

    # ── Row 2: Stochastic ───────────────────────────────────────────────────
    if "stoch_k" in df.columns:
        fig2.add_trace(
            go.Scatter(x=df.index, y=df["stoch_k"], name="%K",
                       line=dict(color="blue", width=1.5)),
            row=2, col=1,
        )
    if "stoch_d" in df.columns:
        fig2.add_trace(
            go.Scatter(x=df.index, y=df["stoch_d"], name="%D",
                       line=dict(color="orange", width=1.5)),
            row=2, col=1,
        )
    fig2.add_hrect(y0=0, y1=20, fillcolor="green", opacity=0.07, line_width=0, row=2, col=1)
    fig2.add_hrect(y0=80, y1=100, fillcolor="red", opacity=0.07, line_width=0, row=2, col=1)
    fig2.add_hline(y=20, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
    fig2.add_hline(y=80, line_dash="dash", line_color="red", line_width=1, row=2, col=1)

    # ── Row 3: MFI ──────────────────────────────────────────────────────────
    if "mfi_14" in df.columns:
        fig2.add_trace(
            go.Scatter(x=df.index, y=df["mfi_14"], name="MFI 14",
                       line=dict(color="teal", width=1.5)),
            row=3, col=1,
        )
    fig2.add_hrect(y0=0, y1=20, fillcolor="green", opacity=0.07, line_width=0, row=3, col=1)
    fig2.add_hrect(y0=80, y1=100, fillcolor="red", opacity=0.07, line_width=0, row=3, col=1)
    fig2.add_hline(y=20, line_dash="dash", line_color="green", line_width=1, row=3, col=1)
    fig2.add_hline(y=80, line_dash="dash", line_color="red", line_width=1, row=3, col=1)

    # ── Row 4: MACD ─────────────────────────────────────────────────────────
    if "macd_hist" in df.columns:
        hist_colors = ["green" if v >= 0 else "red" for v in df["macd_hist"].fillna(0)]
        fig2.add_trace(
            go.Bar(x=df.index, y=df["macd_hist"], name="MACD Hist",
                   marker_color=hist_colors),
            row=4, col=1,
        )
    if "macd" in df.columns:
        fig2.add_trace(
            go.Scatter(x=df.index, y=df["macd"], name="MACD",
                       line=dict(color="blue", width=1.5)),
            row=4, col=1,
        )
    if "macd_signal" in df.columns:
        fig2.add_trace(
            go.Scatter(x=df.index, y=df["macd_signal"], name="Signal Line",
                       line=dict(color="orange", width=1.5)),
            row=4, col=1,
        )
    fig2.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)

    # ── Row 5: Aroon ────────────────────────────────────────────────────────
    if "aroon_up" in df.columns:
        fig2.add_trace(
            go.Scatter(x=df.index, y=df["aroon_up"], name="Aroon Up",
                       line=dict(color="green", width=1.5)),
            row=5, col=1,
        )
    if "aroon_down" in df.columns:
        fig2.add_trace(
            go.Scatter(x=df.index, y=df["aroon_down"], name="Aroon Down",
                       line=dict(color="red", width=1.5)),
            row=5, col=1,
        )
    fig2.add_hline(y=70, line_dash="dash", line_color="gray", line_width=1, row=5, col=1)

    fig2.update_layout(
        hovermode="x unified",
        height=1100,
        title=f"{ticker} — Oscillators ({interval})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Signal Performance
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("Signal Performance vs Buy-and-Hold")
    st.caption("Sorted by Sharpe ratio (descending). Win rate and trade count are not applicable for buy-and-hold.")

    # ── Compute buy-and-hold stats ──────────────────────────────────────────
    bh_rets = df["close"].pct_change()
    bh_cumret = (1.0 + bh_rets).cumprod()
    bh_cumret_clean = bh_cumret.dropna()

    if not bh_cumret_clean.empty:
        bh_total_ret = float(bh_cumret_clean.iloc[-1]) - 1.0
        years = len(df) / 252
        bh_ann_ret = (1.0 + bh_total_ret) ** (1.0 / max(years, 1e-6)) - 1.0
        bh_ann_vol = float(bh_rets.std() * np.sqrt(252))
        bh_sharpe = bh_ann_ret / bh_ann_vol if bh_ann_vol > 0 else np.nan
        bh_roll_max = bh_cumret.cummax()
        bh_mdd = float(((bh_cumret - bh_roll_max) / bh_roll_max).min())
    else:
        bh_total_ret = bh_ann_ret = bh_ann_vol = bh_sharpe = bh_mdd = np.nan

    bh_row = pd.DataFrame(
        [{
            "total_return": round(bh_total_ret, 4) if not np.isnan(bh_total_ret) else np.nan,
            "ann_return":   round(bh_ann_ret, 4)   if not np.isnan(bh_ann_ret) else np.nan,
            "ann_vol":      round(bh_ann_vol, 4)   if not np.isnan(bh_ann_vol) else np.nan,
            "sharpe":       round(bh_sharpe, 4)    if not np.isnan(bh_sharpe) else np.nan,
            "max_drawdown": round(bh_mdd, 4)       if not np.isnan(bh_mdd) else np.nan,
            "win_rate":     np.nan,
            "n_trades":     np.nan,
        }],
        index=pd.Index(["buy_hold"], name="signal"),
    )

    combined = pd.concat([summary_df, bh_row]).sort_values("sharpe", ascending=False)

    # ── Colour helper: green positive, red negative ─────────────────────────
    def _color_fn(val):
        try:
            v = float(val)
            if v > 0:
                return "color: green"
            if v < 0:
                return "color: red"
        except (TypeError, ValueError):
            pass
        return ""

    pct_cols = [c for c in ["total_return", "ann_return", "ann_vol", "max_drawdown", "win_rate"]
                if c in combined.columns]
    fmt = {c: "{:.2%}" for c in pct_cols}
    if "sharpe" in combined.columns:
        fmt["sharpe"] = "{:.3f}"
    if "n_trades" in combined.columns:
        fmt["n_trades"] = "{:.0f}"

    styled = combined.style.format(fmt, na_rep="—")

    color_subset = [c for c in ["total_return", "ann_return"] if c in combined.columns]
    if color_subset:
        try:
            styled = styled.map(_color_fn, subset=color_subset)
        except AttributeError:
            # pandas < 2.1 used applymap
            styled = styled.applymap(_color_fn, subset=color_subset)

    st.dataframe(styled, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Equity Curves
# ---------------------------------------------------------------------------

with tab4:
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3

    fig4 = go.Figure()

    # Buy-and-hold reference (black dashed)
    fig4.add_trace(
        go.Scatter(
            x=df.index,
            y=(1.0 + df["close"].pct_change()).cumprod(),
            name="Buy & Hold",
            line=dict(color="black", width=2.5, dash="dash"),
        )
    )

    # 16 signal equity curves
    for i, name in enumerate(SIGNAL_NAMES):
        cum_col = f"cumret_{name}"
        if cum_col in df.columns:
            fig4.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[cum_col],
                    name=name,
                    line=dict(color=palette[i % len(palette)], width=1.2),
                )
            )

    # Breakeven reference line
    fig4.add_hline(y=1.0, line_dash="dot", line_color="black", line_width=1)

    fig4.update_layout(
        hovermode="x unified",
        height=650,
        title=f"{ticker} — Equity Curves ({interval})",
        yaxis_title="Cumulative Return (1.0 = start)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )

    st.plotly_chart(fig4, use_container_width=True)
