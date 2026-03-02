"""
mean_reversion_dashboard.py
===========================
Streamlit dashboard for the Mean Reversion + Geopolitical Alpha Decay strategy.

Five tabs:
  1. Strategy Backtest      – candlestick with entry/exit markers + equity curves
  2. Half-Life Analysis     – OU half-life + z-score diagnostics
  3. Event Windows          – pre vs post performance for each geopolitical event
  4. Alpha Decay Curves     – day-by-day normalised Sharpe decay + rate of change
  5. Event Heatmap          – cross-event performance heat map

Run:
    streamlit run mean_reversion_dashboard.py
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

from data_ingestion import fetch_ohlcv, FMPAuthError
from mean_reversion import build_mr_strategy, rolling_strategy_metrics, calc_half_life
from geo_events import (
    GEO_EVENTS,
    get_events_in_range,
    calc_alpha_decay,
    build_event_summary,
)

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Mean Reversion + Geo Alpha Decay",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Block 2 — Cached helpers  (scalar args only → reliable cache keys)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _fetch_data(symbol: str, start: str, end: str, interval: str, api_key: str):
    """Fetch OHLCV from FMP; cached 1 hour per unique parameter set."""
    return fetch_ohlcv(symbol, start, end, interval, api_key=api_key or None)


@st.cache_data(ttl=3600)
def _build_strategy(
    symbol: str,
    start: str,
    end: str,
    interval: str,
    api_key: str,
    zscore_window: int,
    entry_z: float,
    exit_z: float,
    allow_short: bool,
    use_ema: bool,
):
    """Fetch + full mean-reversion pipeline; cached 1 hour."""
    df = fetch_ohlcv(symbol, start, end, interval, api_key=api_key or None)
    df, stats_df = build_mr_strategy(df, zscore_window, entry_z, exit_z, allow_short, use_ema)
    return df, stats_df


# ---------------------------------------------------------------------------
# Block 3 — Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Mean Reversion Dashboard")
    st.markdown("---")

    api_key = st.text_input(
        "FMP API Key",
        type="password",
        value=os.environ.get("FMP_API_KEY", ""),
        help="Free key at financialmodelingprep.com",
    )

    ticker = st.text_input(
        "Ticker",
        value="GLD",
        help="Any FMP symbol — equities, ETFs, commodities",
    )

    interval = st.selectbox(
        "Interval",
        ["1day", "1hour", "4hour", "30min", "15min"],
        index=0,
    )

    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start", value=pd.Timestamp("2013-01-01"))
    with col_e:
        end_date = st.date_input("End", value=pd.Timestamp("2024-12-31"))

    st.markdown("##### Strategy")

    zscore_window = st.slider("Z-Score Window", 10, 100, 20, step=5)
    entry_z = st.slider("Entry |z|", 0.5, 4.0, 1.5, step=0.1)
    exit_z  = st.slider("Exit  |z|", 0.0, 2.0, 0.5, step=0.1)

    allow_short = st.checkbox("Allow Short Positions", value=False)
    use_ema     = st.checkbox("Use EMA Z-Score (vs SMA)", value=False)

    st.markdown("##### Event Analysis")

    pre_days     = st.slider("Pre-Event Window (bars)",  10, 120, 60, step=5)
    post_days    = st.slider("Post-Event Window (bars)", 10, 180, 90, step=5)
    decay_window = st.slider("Decay Rolling Window",      5,  40, 15, step=5)

    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Block 4 — Run trigger → session state
# ---------------------------------------------------------------------------

if run_btn:
    errors = []
    if not ticker.strip():
        errors.append("Ticker cannot be empty.")
    if start_date >= end_date:
        errors.append("Start date must be before end date.")
    if entry_z <= exit_z:
        errors.append(f"Entry |z| ({entry_z}) must be greater than Exit |z| ({exit_z}).")

    if errors:
        for e in errors:
            st.sidebar.error(e)
    else:
        with st.spinner(f"Fetching {ticker.strip().upper()} and running strategy…"):
            try:
                df, stats = _build_strategy(
                    ticker.strip().upper(),
                    str(start_date),
                    str(end_date),
                    interval,
                    api_key.strip(),
                    zscore_window,
                    entry_z,
                    float(exit_z),
                    allow_short,
                    use_ema,
                )
                if df.empty:
                    st.warning("No data returned for the requested parameters.")
                else:
                    roll_metrics = rolling_strategy_metrics(
                        df, "signal_mr", "ret_mr", decay_window
                    )
                    st.session_state.update({
                        "df":           df,
                        "stats":        stats,
                        "roll_metrics": roll_metrics,
                        "ticker":       ticker.strip().upper(),
                        "interval":     interval,
                        "zscore_window": zscore_window,
                        "entry_z":       entry_z,
                        "exit_z":        float(exit_z),
                        "pre_days":      pre_days,
                        "post_days":     post_days,
                        "decay_window":  decay_window,
                    })
            except FMPAuthError as exc:
                st.error(f"Authentication error: {exc}")
            except Exception as exc:
                st.error(f"Error: {exc}")

if "df" not in st.session_state:
    st.info(
        "Configure parameters in the sidebar and click **Run Analysis** to get started.\n\n"
        "Default: **GLD** (gold ETF) 2013–2024 with a 20-bar z-score window."
    )
    st.stop()

# Unpack session state
df:           pd.DataFrame = st.session_state["df"]
stats:        pd.DataFrame = st.session_state["stats"]
roll_metrics: pd.DataFrame = st.session_state["roll_metrics"]
ticker:       str          = st.session_state["ticker"]
interval:     str          = st.session_state["interval"]
zscore_window: int         = st.session_state["zscore_window"]
entry_z:      float        = st.session_state["entry_z"]
exit_z:       float        = st.session_state["exit_z"]
pre_days:     int          = st.session_state["pre_days"]
post_days:    int          = st.session_state["post_days"]
decay_window: int          = st.session_state["decay_window"]

st.title(f"{ticker} — Mean Reversion + Geopolitical Alpha Decay")
st.caption(
    f"Interval: {interval}  |  "
    f"{df.index[0].date()} → {df.index[-1].date()}  |  "
    f"{len(df):,} bars  |  "
    f"Z-window: {zscore_window}  |  Entry |z|: {entry_z}  |  Exit |z|: {exit_z}"
)

# Events within data range (used across tabs)
active_events = get_events_in_range(
    str(df.index[0].date()), str(df.index[-1].date())
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Strategy Backtest",
    "📐 Half-Life Analysis",
    "🌍 Event Windows",
    "⚡ Alpha Decay Curves",
    "🔥 Event Heatmap",
])


# ---------------------------------------------------------------------------
# Shared colour helper (pandas 2.1 compat)
# ---------------------------------------------------------------------------

def _color_pct(val):
    try:
        v = float(val)
        if v > 0:
            return "color: green"
        if v < 0:
            return "color: red"
    except (TypeError, ValueError):
        pass
    return ""


def _apply_color(styled, subset):
    try:
        return styled.map(_color_pct, subset=subset)
    except AttributeError:
        return styled.applymap(_color_pct, subset=subset)


# ===========================================================================
# Tab 1 — Strategy Backtest
# ===========================================================================

with tab1:
    # ── Summary metrics ──────────────────────────────────────────────────────
    mr_row = stats.loc["mean_reversion"] if "mean_reversion" in stats.index else None
    bh_row = stats.loc["buy_and_hold"]   if "buy_and_hold"   in stats.index else None

    c1, c2, c3, c4, c5 = st.columns(5)
    if mr_row is not None:
        c1.metric("Total Return",  f"{mr_row['total_return']:.2%}")
        c2.metric("Ann. Return",   f"{mr_row['ann_return']:.2%}")
        c3.metric("Sharpe",        f"{mr_row['sharpe']:.3f}" if not np.isnan(mr_row['sharpe']) else "—")
        c4.metric("Max Drawdown",  f"{mr_row['max_drawdown']:.2%}")
        c5.metric("Win Rate",      f"{mr_row['win_rate']:.2%}"  if not np.isnan(mr_row['win_rate']) else "—")

    # ── Entry / exit marker detection ────────────────────────────────────────
    sig     = df["signal_mr"]
    prev    = sig.shift(1).fillna(0)
    long_entries  = df[(prev != 1)  & (sig == 1)]
    long_exits    = df[(prev == 1)  & (sig != 1)]
    short_entries = df[(prev != -1) & (sig == -1)] if (sig == -1).any() else df.iloc[:0]

    # ── 3-row subplot ────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.50, 0.25, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Price & Signals", "Z-Score", "Cumulative Returns"],
    )

    # Row 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="green", decreasing_line_color="red",
    ), row=1, col=1)

    # Long entry markers ▲
    if len(long_entries) > 0:
        fig.add_trace(go.Scatter(
            x=long_entries.index, y=long_entries["low"] * 0.997,
            mode="markers", marker=dict(symbol="triangle-up", color="limegreen", size=8),
            name="Long Entry",
        ), row=1, col=1)

    # Long exit markers ▼
    if len(long_exits) > 0:
        fig.add_trace(go.Scatter(
            x=long_exits.index, y=long_exits["high"] * 1.003,
            mode="markers", marker=dict(symbol="triangle-down", color="orange", size=8),
            name="Long Exit",
        ), row=1, col=1)

    # Short entry markers ▼ (red)
    if len(short_entries) > 0:
        fig.add_trace(go.Scatter(
            x=short_entries.index, y=short_entries["high"] * 1.003,
            mode="markers", marker=dict(symbol="triangle-down", color="red", size=8),
            name="Short Entry",
        ), row=1, col=1)

    # Row 2: Z-Score
    fig.add_trace(go.Scatter(
        x=df.index, y=df["zscore"], name="Z-Score",
        line=dict(color="steelblue", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y= entry_z, line_dash="dash", line_color="red",   line_width=1.2, row=2, col=1)
    fig.add_hline(y=-entry_z, line_dash="dash", line_color="green", line_width=1.2, row=2, col=1)
    fig.add_hline(y= exit_z,  line_dash="dot",  line_color="orange",line_width=1.0, row=2, col=1)
    fig.add_hline(y=-exit_z,  line_dash="dot",  line_color="orange",line_width=1.0, row=2, col=1)
    fig.add_hrect(y0=-entry_z, y1=-exit_z, fillcolor="green", opacity=0.06, line_width=0, row=2, col=1)
    fig.add_hrect(y0= exit_z,  y1= entry_z, fillcolor="red",  opacity=0.06, line_width=0, row=2, col=1)

    # Row 3: Cumulative returns
    fig.add_trace(go.Scatter(
        x=df.index, y=df["cumret_mr"],  name="MR Strategy",
        line=dict(color="royalblue", width=2),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bh_cumret"], name="Buy & Hold",
        line=dict(color="black", width=1.5, dash="dash"),
    ), row=3, col=1)
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", line_width=1, row=3, col=1)

    # Event vlines (purple dotted) — applied to all subplots (shared x-axis)
    for evt_name, evt in active_events.items():
        evt_ts = pd.Timestamp(evt["date"])
        after  = df.index[df.index >= evt_ts]
        if after.empty:
            continue
        trading_day = after[0]
        fig.add_vline(
            x=trading_day, line_dash="dot",
            line_color="mediumpurple", line_width=1,
        )

    fig.update_layout(
        height=850,
        hovermode="x unified",
        title=f"{ticker} — Mean Reversion Backtest",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Stats table ───────────────────────────────────────────────────────────
    st.subheader("Strategy vs Buy-and-Hold")
    pct_cols = ["total_return", "ann_return", "ann_vol", "max_drawdown", "win_rate", "exposure"]
    fmt_dict = {c: "{:.2%}" for c in pct_cols if c in stats.columns}
    fmt_dict["sharpe"] = "{:.3f}"
    styled = stats.style.format(fmt_dict, na_rep="—")
    styled = _apply_color(styled, subset=[c for c in ["total_return", "ann_return"] if c in stats.columns])
    st.dataframe(styled, use_container_width=True)


# ===========================================================================
# Tab 2 — Half-Life Analysis
# ===========================================================================

with tab2:
    overall_hl = calc_half_life(df["close"])
    reversion_speed = 1.0 / overall_hl if (overall_hl and not np.isnan(overall_hl) and overall_hl > 0) else np.nan

    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Half-Life (days)", f"{overall_hl:.1f}" if not np.isnan(overall_hl) else "—")
    m2.metric("Z-Score Window",           f"{zscore_window} bars")
    m3.metric("Reversion Speed (1/HL)",   f"{reversion_speed:.4f}" if not np.isnan(reversion_speed) else "—")

    st.info(
        "**Ornstein-Uhlenbeck half-life** measures how quickly the price reverts to its mean.  \n"
        "OLS fit: ΔP_t = α + β·P_{t−1}  →  half_life = −ln(2)/β  \n"
        "A negative β indicates mean-reverting behaviour. Shorter half-life = faster reversion."
    )

    # ── 2×2 subplots ─────────────────────────────────────────────────────────
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Rolling Half-Life (60-bar window)",
            "Z-Score Distribution",
            "Z-Score Over Time",
            f"Price vs SMA({zscore_window})",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # TL: Rolling half-life
    rl_hl = df["rolling_hl"].dropna()
    fig2.add_trace(go.Scatter(
        x=rl_hl.index, y=rl_hl.values, name="Rolling HL",
        line=dict(color="darkorange", width=1.5),
    ), row=1, col=1)
    if not np.isnan(overall_hl):
        fig2.add_hline(
            y=overall_hl, line_dash="dash", line_color="red", line_width=1.5,
            annotation_text=f"Overall HL: {overall_hl:.1f}d", row=1, col=1,
        )

    # TR: Z-score histogram
    z_vals = df["zscore"].dropna()
    fig2.add_trace(go.Histogram(
        x=z_vals, name="Z-Score Dist",
        marker_color="steelblue", opacity=0.7, nbinsx=60,
    ), row=1, col=2)
    fig2.add_vline(x= entry_z, line_dash="dash", line_color="red",   row=1, col=2)
    fig2.add_vline(x=-entry_z, line_dash="dash", line_color="green", row=1, col=2)

    # BL: Z-score over time with shaded threshold zone
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["zscore"], name="Z-Score",
        line=dict(color="steelblue", width=1), showlegend=False,
    ), row=2, col=1)
    fig2.add_hrect(y0=-entry_z, y1=entry_z, fillcolor="lightblue",
                   opacity=0.15, line_width=0, row=2, col=1)
    fig2.add_hline(y=0, line_color="gray", line_width=1, row=2, col=1)

    # BR: Price vs rolling SMA
    sma_col = f"sma_{zscore_window}" if f"sma_{zscore_window}" in df.columns else None
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["close"], name="Close",
        line=dict(color="black", width=1.2), showlegend=False,
    ), row=2, col=2)
    # Compute SMA on the fly if not in df
    sma_vals = df["close"].rolling(zscore_window).mean()
    fig2.add_trace(go.Scatter(
        x=df.index, y=sma_vals, name=f"SMA({zscore_window})",
        line=dict(color="royalblue", width=1.5, dash="dash"), showlegend=False,
    ), row=2, col=2)

    fig2.update_layout(
        height=750, hovermode="x unified",
        title=f"{ticker} — Half-Life & Z-Score Diagnostics",
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Z-score summary stats ─────────────────────────────────────────────────
    z_clean = df["zscore"].dropna()
    if len(z_clean) > 0:
        pct_outside = float((z_clean.abs() > entry_z).mean())
        z_stats = pd.DataFrame({
            "Metric": ["Mean", "Std Dev", "Skewness", "Kurtosis", f"% Outside ±{entry_z}"],
            "Value":  [
                f"{z_clean.mean():.4f}",
                f"{z_clean.std():.4f}",
                f"{z_clean.skew():.4f}",
                f"{z_clean.kurtosis():.4f}",
                f"{pct_outside:.2%}",
            ],
        })
        st.dataframe(z_stats, use_container_width=True, hide_index=True)


# ===========================================================================
# Tab 3 — Event Windows
# ===========================================================================

with tab3:
    st.subheader("Geopolitical Event Catalog")

    # Full catalog table (all 15 events)
    catalog_rows = []
    for name, evt in GEO_EVENTS.items():
        catalog_rows.append({
            "Event":           name,
            "Date":            evt["date"],
            "Category":        evt["category"],
            "Region":          evt["region"],
            "Expected Impact": evt["expected_impact"],
            "In Range":        "✓" if name in active_events else "",
        })
    catalog_df = pd.DataFrame(catalog_rows)
    st.dataframe(catalog_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Pre vs Post Performance (Events in Data Range)")

    if not active_events:
        st.warning("No events fall within the selected date range.")
    else:
        with st.spinner("Computing event window statistics…"):
            summary_df = build_event_summary(
                df, active_events, "signal_mr", "ret_mr", pre_days, post_days
            )

        if not summary_df.empty:
            # Format the display columns
            display = summary_df[[
                "event", "date", "category",
                "pre_sharpe", "post_sharpe", "alpha_change_sharpe",
                "pre_hit_rate", "post_hit_rate", "alpha_change_hit_rate",
                "post_exposure",
            ]].copy()
            display["date"] = display["date"].dt.strftime("%Y-%m-%d")

            fmt = {
                "pre_sharpe":            "{:.3f}",
                "post_sharpe":           "{:.3f}",
                "alpha_change_sharpe":   "{:.3f}",
                "pre_hit_rate":          "{:.2%}",
                "post_hit_rate":         "{:.2%}",
                "alpha_change_hit_rate": "{:.2%}",
                "post_exposure":         "{:.2%}",
            }
            styled_summ = display.style.format(fmt, na_rep="—")
            styled_summ = _apply_color(
                styled_summ,
                subset=["alpha_change_sharpe", "alpha_change_hit_rate"],
            )
            st.dataframe(styled_summ, use_container_width=True, hide_index=True)

            # Horizontal bar chart — alpha change in Sharpe
            chart_df = summary_df[["event", "alpha_change_sharpe"]].dropna(subset=["alpha_change_sharpe"])
            chart_df = chart_df.sort_values("alpha_change_sharpe", ascending=True)

            fig3b = go.Figure(go.Bar(
                x=chart_df["alpha_change_sharpe"],
                y=chart_df["event"],
                orientation="h",
                marker_color=[
                    "green" if v >= 0 else "red"
                    for v in chart_df["alpha_change_sharpe"]
                ],
                text=[f"{v:+.3f}" for v in chart_df["alpha_change_sharpe"]],
                textposition="outside",
            ))
            fig3b.add_vline(x=0, line_color="black", line_width=1.5)
            fig3b.update_layout(
                height=max(350, len(chart_df) * 30),
                title="Alpha Change in Sharpe Ratio (post vs pre event, normalised)",
                xaxis_title="(Post Sharpe − Pre Sharpe) / |Pre Sharpe|",
                yaxis_title="",
                showlegend=False,
            )
            st.plotly_chart(fig3b, use_container_width=True)


# ===========================================================================
# Tab 4 — Alpha Decay Curves  (primary deliverable)
# ===========================================================================

with tab4:
    if not active_events:
        st.warning("No events fall within the selected date range.")
    else:
        # Event selector
        event_names = list(active_events.keys())
        selected_event = st.selectbox("Select Event", event_names, index=0)
        evt_meta = active_events[selected_event]

        # Event metadata row
        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(f"**Date:** {evt_meta['date']}")
        mc2.markdown(f"**Category:** {evt_meta['category']}")
        mc3.markdown(f"**Expected Impact:** {evt_meta['expected_impact']}")

        with st.spinner("Computing alpha decay curve…"):
            decay_df = calc_alpha_decay(
                df, evt_meta["date"], "signal_mr", "ret_mr",
                post_days=post_days, rolling_window=decay_window, pre_days=pre_days,
            )

        if decay_df.empty:
            st.warning("Insufficient data to compute alpha decay for this event.")
        else:
            baseline_sharpe = decay_df.attrs.get("baseline_sharpe", np.nan)

            # Recovery metrics
            valid_norm = decay_df["norm_sharpe"].dropna()
            min_norm   = float(valid_norm.min())  if len(valid_norm) > 0 else np.nan
            recovery_90 = np.nan
            if len(valid_norm) > 0:
                recovery_mask = valid_norm[valid_norm >= 0.90]
                if not recovery_mask.empty:
                    recovery_90 = int(recovery_mask.index[0])  # day_offset value

            max_roc = float(decay_df["rate_of_change"].abs().dropna().max()) if len(decay_df) > 0 else np.nan

            dm1, dm2, dm3, dm4 = st.columns(4)
            dm1.metric("Baseline Sharpe",       f"{baseline_sharpe:.3f}" if not np.isnan(baseline_sharpe) else "—")
            dm2.metric("Min Normalised Sharpe",  f"{min_norm:.3f}"        if not np.isnan(min_norm)        else "—")
            dm3.metric("Days to 90% Recovery",   f"{recovery_90}"         if not np.isnan(recovery_90)     else "—")
            dm4.metric("Max |Rate of Change|",   f"{max_roc:.3f}"         if not np.isnan(max_roc)         else "—")

            # ── 3-row subplot ─────────────────────────────────────────────────
            fig4 = make_subplots(
                rows=3, cols=1,
                row_heights=[0.40, 0.30, 0.30],
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=[
                    "Normalised Rolling Sharpe",
                    "Rolling Hit Rate",
                    "Rate of Change (Daily Alpha Decay)",
                ],
            )

            x = decay_df["day_offset"]

            # Row 1: norm_sharpe
            fig4.add_trace(go.Scatter(
                x=x, y=decay_df["norm_sharpe"], name="Norm. Sharpe",
                line=dict(color="royalblue", width=2.5),
            ), row=1, col=1)
            fig4.add_hline(y=1.0,  line_dash="dash", line_color="green",  line_width=1.5,
                           annotation_text="Baseline", row=1, col=1)
            fig4.add_hline(y=0.90, line_dash="dot",  line_color="orange", line_width=1.2,
                           annotation_text="90%",      row=1, col=1)
            fig4.add_hline(y=0.0,  line_dash="dot",  line_color="red",    line_width=1.0, row=1, col=1)
            fig4.add_vline(x=0, line_dash="dash", line_color="mediumpurple", line_width=1.5, row=1, col=1)

            # Row 2: hit_rate
            fig4.add_trace(go.Scatter(
                x=x, y=decay_df["hit_rate"], name="Hit Rate",
                line=dict(color="teal", width=2),
            ), row=2, col=1)
            fig4.add_hline(y=0.50, line_dash="dash", line_color="gray",         line_width=1.0, row=2, col=1)
            fig4.add_vline(x=0,    line_dash="dash", line_color="mediumpurple", line_width=1.5, row=2, col=1)

            # Row 3: rate_of_change bars (green ≥ 0, red < 0)
            roc = decay_df["rate_of_change"].fillna(0)
            bar_colors = ["green" if v >= 0 else "red" for v in roc]
            fig4.add_trace(go.Bar(
                x=x, y=roc, name="RoC",
                marker_color=bar_colors, showlegend=False,
            ), row=3, col=1)
            fig4.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
            fig4.add_vline(x=0, line_dash="dash", line_color="mediumpurple", line_width=1.5, row=3, col=1)

            fig4.update_xaxes(title_text="Days Since Event (0 = event day)", row=3, col=1)
            fig4.update_layout(
                height=750,
                title=f"Alpha Decay: {selected_event} ({evt_meta['date']})",
                hovermode="x unified",
                showlegend=True,
            )
            st.plotly_chart(fig4, use_container_width=True)

            with st.expander("Raw Decay Data"):
                st.dataframe(
                    decay_df.style.format({
                        "rolling_sharpe": "{:.4f}",
                        "hit_rate":       "{:.4f}",
                        "avg_ret":        "{:.6f}",
                        "norm_sharpe":    "{:.4f}",
                        "rate_of_change": "{:.4f}",
                    }, na_rep="—"),
                    use_container_width=True,
                )


# ===========================================================================
# Tab 5 — Event Heatmap
# ===========================================================================

with tab5:
    if not active_events:
        st.warning("No events fall within the selected date range.")
    else:
        with st.spinner("Building event heatmap…"):
            summary_df5 = build_event_summary(
                df, active_events, "signal_mr", "ret_mr", pre_days, post_days
            )

        if summary_df5.empty:
            st.warning("No event data available for heatmap.")
        else:
            hmap_cols = {
                "Sharpe Change":    "alpha_change_sharpe",
                "Hit Rate Change":  "alpha_change_hit_rate",
                "Post Sharpe":      "post_sharpe",
                "Pre Sharpe":       "pre_sharpe",
                "Post Exposure":    "post_exposure",
            }

            z_matrix   = []
            text_matrix = []
            for col_label, col_key in hmap_cols.items():
                col_vals  = summary_df5[col_key].fillna(0).tolist()
                col_texts = [f"{v:.3f}" for v in col_vals]
                z_matrix.append(col_vals)
                text_matrix.append(col_texts)

            event_labels = summary_df5["event"].tolist()

            fig5 = go.Figure(go.Heatmap(
                z=z_matrix,
                x=event_labels,
                y=list(hmap_cols.keys()),
                colorscale="RdYlGn",
                zmid=0,
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=10),
                colorbar=dict(title="Value"),
            ))
            fig5.update_layout(
                height=max(350, len(hmap_cols) * 70),
                title=f"{ticker} — Geopolitical Event Performance Heatmap",
                xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
            )
            st.plotly_chart(fig5, use_container_width=True)

            # ── Most / Least Damaging ─────────────────────────────────────────
            valid_alpha = summary_df5.dropna(subset=["alpha_change_sharpe"]).copy()
            if len(valid_alpha) >= 2:
                most_damaging  = valid_alpha.nsmallest(3, "alpha_change_sharpe")
                least_damaging = valid_alpha.nlargest(3, "alpha_change_sharpe")

                col_dam, col_rec = st.columns(2)
                with col_dam:
                    st.markdown("**Most Damaging Events** (largest negative Sharpe change)")
                    for _, row in most_damaging.iterrows():
                        st.markdown(
                            f"- **{row['event']}** ({row['date'].strftime('%Y-%m-%d')})  "
                            f"Δ Sharpe: `{row['alpha_change_sharpe']:.3f}`"
                        )
                with col_rec:
                    st.markdown("**Least Damaging / Beneficial Events** (largest positive Sharpe change)")
                    for _, row in least_damaging.iterrows():
                        st.markdown(
                            f"- **{row['event']}** ({row['date'].strftime('%Y-%m-%d')})  "
                            f"Δ Sharpe: `{row['alpha_change_sharpe']:.3f}`"
                        )

            # ── Rolling Sharpe time series with event vlines ──────────────────
            st.markdown("---")
            st.subheader("Rolling Sharpe Over Time with Event Markers")

            palette = px.colors.qualitative.Set1
            fig5b = go.Figure()
            fig5b.add_trace(go.Scatter(
                x=roll_metrics.index,
                y=roll_metrics["rolling_sharpe"],
                name="Rolling Sharpe",
                line=dict(color="royalblue", width=1.8),
            ))
            fig5b.add_hline(y=0, line_color="black", line_width=1)

            for idx, (evt_name, evt) in enumerate(active_events.items()):
                evt_ts = pd.Timestamp(evt["date"])
                after  = df.index[df.index >= evt_ts]
                if after.empty:
                    continue
                trading_day = after[0]
                color = palette[idx % len(palette)]
                fig5b.add_vline(
                    x=trading_day,
                    line_dash="dash",
                    line_color=color,
                    line_width=1.5,
                    annotation_text=evt_name[:20],
                    annotation_font_size=9,
                    annotation_textangle=-90,
                )

            fig5b.update_layout(
                height=450,
                hovermode="x unified",
                title=f"{ticker} — Rolling {decay_window}-bar Sharpe with Geopolitical Events",
                yaxis_title="Rolling Sharpe",
                showlegend=True,
            )
            st.plotly_chart(fig5b, use_container_width=True)
