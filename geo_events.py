"""
geo_events.py
=============
Geopolitical event catalog and alpha decay analysis.

Provides a catalog of 15 major geopolitical events (2014–2024) and functions
to measure how mean-reversion strategy alpha changes around those events —
specifically the day-over-day rate of change in normalised rolling Sharpe.

Usage
-----
    from geo_events import GEO_EVENTS, get_events_in_range, calc_alpha_decay, build_event_summary

    events = get_events_in_range("2019-01-01", "2024-01-01")
    decay_df = calc_alpha_decay(df, "2020-03-11", "signal_mr", "ret_mr")
    summary  = build_event_summary(df, events, "signal_mr", "ret_mr")
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from mean_reversion import rolling_strategy_metrics, TRADING_DAYS_PER_YEAR

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Geopolitical event catalog (15 events, 2014–2024)
# ---------------------------------------------------------------------------

GEO_EVENTS: Dict[str, dict] = {
    "Crimea Annexation": {
        "date": "2014-03-18",
        "description": "Russia formally annexes Crimea from Ukraine",
        "category": "War",
        "region": "Europe",
        "expected_impact": "Sanctions regime, energy supply concern",
    },
    "Brexit Referendum": {
        "date": "2016-06-24",
        "description": "UK votes to leave the European Union",
        "category": "Political Shock",
        "region": "Europe",
        "expected_impact": "GBP collapse, European equity sell-off",
    },
    "US-China Tariffs Round 1": {
        "date": "2018-03-22",
        "description": "Trump announces $60B tariffs on Chinese goods",
        "category": "Trade War",
        "region": "Global",
        "expected_impact": "Market uncertainty, equity dip",
    },
    "Iran Nuclear Deal Exit": {
        "date": "2018-05-08",
        "description": "US withdraws from Iran nuclear deal (JCPOA)",
        "category": "Political Shock",
        "region": "Middle East",
        "expected_impact": "Iran sanctions, oil price impact",
    },
    "US-China Tariffs Escalation": {
        "date": "2019-05-13",
        "description": "US raises tariffs to 25% on $200B Chinese goods",
        "category": "Trade War",
        "region": "Global",
        "expected_impact": "Equity sell-off, safe-haven demand",
    },
    "Saudi Aramco Attack": {
        "date": "2019-09-14",
        "description": "Drone attack on Saudi Aramco facilities",
        "category": "Infra Attack",
        "region": "Middle East",
        "expected_impact": "Oil supply shock",
    },
    "US-Iran Strike (Soleimani)": {
        "date": "2020-01-03",
        "description": "US drone strike kills Iranian General Soleimani",
        "category": "Military Strike",
        "region": "Middle East",
        "expected_impact": "Oil spike, brief risk-off",
    },
    "COVID-19 Pandemic": {
        "date": "2020-03-11",
        "description": "WHO declared COVID-19 a pandemic",
        "category": "Pandemic",
        "region": "Global",
        "expected_impact": "Extreme volatility spike, equity sell-off",
    },
    "US Capitol Insurrection": {
        "date": "2021-01-06",
        "description": "Capitol Hill stormed, US political uncertainty",
        "category": "Political Shock",
        "region": "USA",
        "expected_impact": "Brief USD weakness, political risk premium",
    },
    "Russia-Ukraine War": {
        "date": "2022-02-24",
        "description": "Russia's full-scale invasion of Ukraine",
        "category": "War",
        "region": "Europe",
        "expected_impact": "Energy price spike, risk-off",
    },
    "Taiwan Strait Crisis": {
        "date": "2022-08-02",
        "description": "Pelosi visits Taiwan, China military exercises",
        "category": "Military Tension",
        "region": "Asia-Pacific",
        "expected_impact": "Semiconductor supply fears, Asia risk-off",
    },
    "Nord Stream Sabotage": {
        "date": "2022-09-26",
        "description": "Nord Stream pipeline explosions",
        "category": "Infra Attack",
        "region": "Europe",
        "expected_impact": "Energy security concerns, European gas spike",
    },
    "SVB Bank Collapse": {
        "date": "2023-03-10",
        "description": "Silicon Valley Bank failure triggers banking fears",
        "category": "Financial Crisis",
        "region": "USA",
        "expected_impact": "Banking sector stress, rate expectations shift",
    },
    "Israel-Hamas War": {
        "date": "2023-10-07",
        "description": "Hamas attack on Israel, start of Gaza conflict",
        "category": "War",
        "region": "Middle East",
        "expected_impact": "Oil spike, regional tension",
    },
    "Houthi Red Sea Attacks": {
        "date": "2023-12-19",
        "description": "Houthi missile attacks disrupt Red Sea shipping",
        "category": "Military Strike",
        "region": "Middle East",
        "expected_impact": "Shipping cost spike, supply chain disruption",
    },
}


# ---------------------------------------------------------------------------
# 2. Range filter
# ---------------------------------------------------------------------------

def get_events_in_range(start: str, end: str) -> Dict[str, dict]:
    """
    Filter GEO_EVENTS to events whose date falls within [start, end].

    Parameters
    ----------
    start : "YYYY-MM-DD"
    end   : "YYYY-MM-DD"

    Returns
    -------
    Subset of GEO_EVENTS dict, preserving original order.
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    return {
        name: evt
        for name, evt in GEO_EVENTS.items()
        if start_ts <= pd.Timestamp(evt["date"]) <= end_ts
    }


# ---------------------------------------------------------------------------
# 3. Event window extraction
# ---------------------------------------------------------------------------

def get_event_window(
    df: pd.DataFrame,
    event_date: str,
    pre_days: int,
    post_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Locate the nearest trading day at/after event_date in df's index.
    Returns (pre_df, post_df), each clamped to df boundaries.

    pre_df  : up to pre_days bars *before* the event bar
    post_df : the event bar and up to post_days bars after it
    """
    event_ts = pd.Timestamp(event_date)

    # Find nearest trading day at or after event date
    after = df.index[df.index >= event_ts]
    if not after.empty:
        event_idx = df.index.get_loc(after[0])
    else:
        # Event is after all data — use last available bar
        before = df.index[df.index <= event_ts]
        if before.empty:
            return df.iloc[:0], df.iloc[:0]
        event_idx = df.index.get_loc(before[-1])

    pre_start = max(0, event_idx - pre_days)
    post_end  = min(len(df), event_idx + post_days + 1)

    pre_df  = df.iloc[pre_start : event_idx]
    post_df = df.iloc[event_idx : post_end]
    return pre_df, post_df


# ---------------------------------------------------------------------------
# 4. Private statistical helpers
# ---------------------------------------------------------------------------

def _calc_sharpe(rets: pd.Series) -> float:
    """Annualised Sharpe ratio from a series of daily returns."""
    rets = rets.dropna()
    if len(rets) < 2:
        return np.nan
    mean_r = rets.mean()
    std_r  = rets.std(ddof=1)
    if std_r <= 0:
        return np.nan
    return float(mean_r / std_r * np.sqrt(TRADING_DAYS_PER_YEAR))


def _calc_hit_rate(rets: pd.Series) -> float:
    """Fraction of bars with positive returns."""
    rets = rets.dropna()
    if len(rets) == 0:
        return np.nan
    return float((rets > 0).mean())


# ---------------------------------------------------------------------------
# 5. Event window statistics
# ---------------------------------------------------------------------------

def calc_event_window_stats(
    df: pd.DataFrame,
    event_date: str,
    signal_col: str,
    ret_col: str,
    pre_days: int = 60,
    post_days: int = 90,
) -> dict:
    """
    Compute pre-event and post-event strategy performance statistics.

    Returns dict with keys:
        pre_sharpe, pre_hit_rate, post_sharpe, post_hit_rate,
        alpha_change_sharpe, alpha_change_hit_rate,
        pre_exposure, post_exposure
    """
    pre_df, post_df = get_event_window(df, event_date, pre_days, post_days)

    pre_rets  = pre_df[ret_col]  if ret_col  in pre_df.columns  and len(pre_df)  > 0 else pd.Series(dtype=float)
    post_rets = post_df[ret_col] if ret_col  in post_df.columns and len(post_df) > 0 else pd.Series(dtype=float)

    pre_sharpe  = _calc_sharpe(pre_rets)
    pre_hit     = _calc_hit_rate(pre_rets)
    post_sharpe = _calc_sharpe(post_rets)
    post_hit    = _calc_hit_rate(post_rets)

    # Normalised alpha change: (post − pre) / |pre|
    def _norm_change(post_val, pre_val):
        if np.isnan(post_val) or np.isnan(pre_val) or abs(pre_val) < 1e-10:
            return np.nan
        return (post_val - pre_val) / abs(pre_val)

    pre_exp  = float((pre_df[signal_col]  != 0).mean()) if signal_col in pre_df.columns  and len(pre_df)  > 0 else np.nan
    post_exp = float((post_df[signal_col] != 0).mean()) if signal_col in post_df.columns and len(post_df) > 0 else np.nan

    return {
        "pre_sharpe":            pre_sharpe,
        "pre_hit_rate":          pre_hit,
        "post_sharpe":           post_sharpe,
        "post_hit_rate":         post_hit,
        "alpha_change_sharpe":   _norm_change(post_sharpe, pre_sharpe),
        "alpha_change_hit_rate": _norm_change(post_hit,    pre_hit),
        "pre_exposure":          pre_exp,
        "post_exposure":         post_exp,
    }


# ---------------------------------------------------------------------------
# 6. Alpha decay curve
# ---------------------------------------------------------------------------

def calc_alpha_decay(
    df: pd.DataFrame,
    event_date: str,
    signal_col: str,
    ret_col: str,
    post_days: int = 90,
    rolling_window: int = 15,
    pre_days: int = 60,
) -> pd.DataFrame:
    """
    Compute the alpha decay curve around a geopolitical event.

    For each day offset [0 … post_days] after the event:
      1. Compute rolling `rolling_window`-bar Sharpe on the full series
      2. Normalise by pre-event baseline → norm_sharpe
      3. rate_of_change = norm_sharpe.diff()

    Columns returned:
        day_offset, rolling_sharpe, hit_rate, avg_ret, norm_sharpe, rate_of_change

    The baseline Sharpe is stored in the DataFrame's attrs dict
    under the key ``"baseline_sharpe"``.

    Returns empty DataFrame if the event date is outside df's range.
    """
    event_ts = pd.Timestamp(event_date)

    # Locate event index
    after = df.index[df.index >= event_ts]
    if after.empty:
        return pd.DataFrame()
    event_idx = df.index.get_loc(after[0])

    # Compute rolling metrics on entire df
    roll = rolling_strategy_metrics(df, signal_col, ret_col, rolling_window)

    # Pre-event baseline Sharpe (mean of rolling Sharpes in pre window)
    pre_start = max(0, event_idx - pre_days)
    pre_sharpes = roll["rolling_sharpe"].iloc[pre_start : event_idx].dropna()
    baseline_sharpe = float(pre_sharpes.mean()) if len(pre_sharpes) > 0 else np.nan

    # Post-event window
    post_end = min(len(df), event_idx + post_days + 1)
    post_roll = roll.iloc[event_idx : post_end].copy()

    if post_roll.empty:
        return pd.DataFrame()

    result = pd.DataFrame({
        "day_offset":      range(len(post_roll)),
        "rolling_sharpe":  post_roll["rolling_sharpe"].values,
        "hit_rate":        post_roll["hit_rate"].values,
        "avg_ret":         post_roll["avg_ret"].values,
    })

    # Normalise by absolute baseline
    if not np.isnan(baseline_sharpe) and abs(baseline_sharpe) > 1e-10:
        result["norm_sharpe"] = result["rolling_sharpe"] / abs(baseline_sharpe)
    else:
        result["norm_sharpe"] = np.nan

    result["rate_of_change"] = result["norm_sharpe"].diff()

    # Store baseline for downstream use
    result.attrs["baseline_sharpe"] = baseline_sharpe

    return result


# ---------------------------------------------------------------------------
# 7. Event summary table
# ---------------------------------------------------------------------------

def build_event_summary(
    df: pd.DataFrame,
    events: Dict[str, dict],
    signal_col: str,
    ret_col: str,
    pre_days: int = 60,
    post_days: int = 90,
) -> pd.DataFrame:
    """
    Aggregate calc_event_window_stats for all events in the dict.

    Returns DataFrame (one row per event) sorted by event date, with columns:
        event, date, category, region, expected_impact,
        pre_sharpe, pre_hit_rate, post_sharpe, post_hit_rate,
        alpha_change_sharpe, alpha_change_hit_rate,
        pre_exposure, post_exposure
    """
    rows = []
    for name, evt in events.items():
        stats = calc_event_window_stats(
            df, evt["date"], signal_col, ret_col, pre_days, post_days
        )
        rows.append({
            "event":           name,
            "date":            evt["date"],
            "category":        evt["category"],
            "region":          evt["region"],
            "expected_impact": evt["expected_impact"],
            **stats,
        })

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    summary["date"] = pd.to_datetime(summary["date"])
    return summary.sort_values("date").reset_index(drop=True)
