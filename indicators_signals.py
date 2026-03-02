"""
indicators_signals.py
=====================
FMP-powered Technical Indicator & Signal Module

Usage
-----
    import os
    os.environ['FMP_API_KEY'] = 'YOUR_KEY'
    from indicators_signals import build_signals_and_returns

    df, summary = build_signals_and_returns('AAPL', '2022-01-01', '2024-01-01')

Design Notes
------------
- Returns: simple returns via pct_change (not log returns)
- EMA: SMA-seeded; alpha = 2/(period+1), first value = SMA of first `period` bars
- RSI: simple rolling mean of gains/losses (not Wilder's EMA)
- No lookahead: position_t is applied to return_{t+1} via .shift(1)
- Signals: 0 (flat) or 1 (long); shorts are not taken
- All signal columns default to 0 (not NaN) where indicators are undefined
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from data_ingestion import fetch_ohlcv as _fetch_ohlcv

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Module-level constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252

SIGNAL_NAMES = [
    "sma20",
    "ema20",
    "rsi_lt30_gt70",
    "rsi_avg_x50",
    "rsi_x_avg",
    "rsi_bb",
    "bb_mid",
    "bb_lower_upper",
    "mfi",
    "stoch_k_lvl",
    "stoch_kd_x",
    "aroon_x",
    "aroon_filtered",
    "macd_x",
    "macd_zero",
    "vwap",
]


# ---------------------------------------------------------------------------
# 2. FMP data fetching (delegates to data_ingestion module)
# ---------------------------------------------------------------------------

def fetch_fmp_ohlcv(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1day",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV from FMP via data_ingestion.fetch_ohlcv."""
    return _fetch_ohlcv(symbol, start, end, interval=interval, api_key=api_key)


# ---------------------------------------------------------------------------
# 3. Indicator helpers
# ---------------------------------------------------------------------------

def _ema_seeded(series: pd.Series, period: int) -> pd.Series:
    """
    SMA-seeded EMA.
    First value = SMA of first `period` non-NaN observations.
    Subsequent values: ema_t = alpha * price_t + (1 - alpha) * ema_{t-1}
    """
    alpha = 2.0 / (period + 1)
    result = np.full(len(series), np.nan)
    values = series.values

    # Find first valid index with enough history
    valid = np.where(~np.isnan(values))[0]
    if len(valid) < period:
        return pd.Series(result, index=series.index)

    seed_end = valid[period - 1]  # index of the period-th valid bar
    # Seed: SMA of first `period` valid bars
    seed_vals = values[valid[:period]]
    result[seed_end] = np.mean(seed_vals)

    # Forward fill EMA from seed_end
    for i in range(seed_end + 1, len(values)):
        if np.isnan(values[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]

    return pd.Series(result, index=series.index)


def calc_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add sma_{period} column."""
    df[f"sma_{period}"] = df["close"].rolling(period).mean()
    return df


def calc_ema(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add ema_{period} column using SMA-seeded EMA."""
    df[f"ema_{period}"] = _ema_seeded(df["close"], period)
    return df


def calc_rsi(
    df: pd.DataFrame, period: int = 14, avg_period: int = 14
) -> pd.DataFrame:
    """
    Add rsi_{period} and rsi_avg_{avg_period} columns.
    Uses simple rolling mean (not Wilder's EMA).
    """
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100.0 - (100.0 / (1.0 + rs))
    df[f"rsi_avg_{avg_period}"] = df[f"rsi_{period}"].rolling(avg_period).mean()
    return df


def calc_rsi_bollinger(
    df: pd.DataFrame,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    """
    Add Bollinger Bands of RSI: rsi_bb_upper, rsi_bb_mid, rsi_bb_lower.
    Requires rsi_{rsi_period} column (calls calc_rsi if missing).
    """
    rsi_col = f"rsi_{rsi_period}"
    if rsi_col not in df.columns:
        df = calc_rsi(df, rsi_period)
    rsi = df[rsi_col]
    mid = rsi.rolling(bb_period).mean()
    std = rsi.rolling(bb_period).std(ddof=0)
    df["rsi_bb_mid"] = mid
    df["rsi_bb_upper"] = mid + bb_std * std
    df["rsi_bb_lower"] = mid - bb_std * std
    return df


def calc_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std: float = 2.0
) -> pd.DataFrame:
    """Add bb_upper, bb_mid, bb_lower columns."""
    mid = df["close"].rolling(period).mean()
    s = df["close"].rolling(period).std(ddof=0)
    df["bb_mid"] = mid
    df["bb_upper"] = mid + std * s
    df["bb_lower"] = mid - std * s
    return df


def calc_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add mfi_{period} column. Requires high, low, close, volume."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].fillna(method="ffill")
    raw_money = tp * vol
    pos_flow = raw_money.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    neg_flow = raw_money.where(tp < tp.shift(1), 0.0).rolling(period).sum()
    mfr = pos_flow / neg_flow.replace(0, np.nan)
    df[f"mfi_{period}"] = 100.0 - (100.0 / (1.0 + mfr))
    return df


def calc_stochastic(
    df: pd.DataFrame, k: int = 14, d: int = 3
) -> pd.DataFrame:
    """Add stoch_k and stoch_d columns."""
    low_min = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df["stoch_k"] = 100.0 * (df["close"] - low_min) / denom
    df["stoch_d"] = df["stoch_k"].rolling(d).mean()
    return df


def calc_aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    """
    Add aroon_up and aroon_down columns.
    Loop-based idxmax/idxmin per rolling window.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)

    for i in range(period, n):
        window_h = highs[i - period : i + 1]
        window_l = lows[i - period : i + 1]
        bars_since_high = period - np.argmax(window_h)
        bars_since_low = period - np.argmin(window_l)
        aroon_up[i] = ((period - bars_since_high) / period) * 100.0
        aroon_down[i] = ((period - bars_since_low) / period) * 100.0

    df["aroon_up"] = aroon_up
    df["aroon_down"] = aroon_down
    return df


def calc_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Add macd, macd_signal, macd_hist columns.
    EMA components are SMA-seeded.
    """
    ema_fast = _ema_seeded(df["close"], fast)
    ema_slow = _ema_seeded(df["close"], slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema_seeded(macd_line, signal)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line
    return df


def calc_vwap(df: pd.DataFrame, interval: str = "1day") -> pd.DataFrame:
    """
    Add vwap_daily_cum column.
    Daily: cumulative VWAP from inception.
    Intraday: resets each calendar day.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].fillna(method="ffill").fillna(0)

    if interval == "1day":
        cum_tpv = (tp * vol).cumsum()
        cum_vol = vol.cumsum().replace(0, np.nan)
        df["vwap_daily_cum"] = cum_tpv / cum_vol
    else:
        date_key = df.index.date
        tpv = tp * vol
        cum_tpv = tpv.groupby(date_key).cumsum()
        cum_vol = vol.groupby(date_key).cumsum().replace(0, np.nan)
        df["vwap_daily_cum"] = cum_tpv / cum_vol
        # Alias for intraday session VWAP
        df["vwap_session"] = df["vwap_daily_cum"]

    return df


def _calc_all_indicators(df: pd.DataFrame, interval: str = "1day") -> pd.DataFrame:
    """Run all indicator calculations in dependency order."""
    df = calc_sma(df, 20)
    df = calc_ema(df, 20)
    df = calc_rsi(df, 14, 14)
    df = calc_rsi_bollinger(df, 14, 20, 2.0)
    df = calc_bollinger_bands(df, 20, 2.0)
    df = calc_mfi(df, 14)
    df = calc_stochastic(df, 14, 3)
    df = calc_aroon(df, 25)
    df = calc_macd(df, 12, 26, 9)
    df = calc_vwap(df, interval)
    return df


# ---------------------------------------------------------------------------
# 4. Signal generation helpers
# ---------------------------------------------------------------------------

def _compare_signal(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    Returns 1 where a > b, 0 otherwise.
    NaN in either series produces 0.
    """
    result = (a > b).astype(int)
    result[a.isna() | b.isna()] = 0
    return result


def _state_signal(enter_cond: pd.Series, exit_cond: pd.Series) -> pd.Series:
    """
    State-machine signal: enter on enter_cond, exit on exit_cond.
    Remains in state until exit_cond fires.
    NaN positions produce state=0.
    """
    enter = enter_cond.fillna(False).values.astype(bool)
    exit_ = exit_cond.fillna(False).values.astype(bool)
    state = np.zeros(len(enter), dtype=int)
    current = 0
    for i in range(len(enter)):
        if current == 0 and enter[i]:
            current = 1
        elif current == 1 and exit_[i]:
            current = 0
        state[i] = current
    return pd.Series(state, index=enter_cond.index)


# ---------------------------------------------------------------------------
# 5. Signal generation (16 signals)
# ---------------------------------------------------------------------------

def generate_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all 16 signal columns. Each is 1 (long) or 0 (flat).
    Signals are generated at bar close; positions take effect next bar.
    """
    close = df["close"]

    # ---- SMA20 ----
    df["signal_sma20"] = _compare_signal(close, df["sma_20"])

    # ---- EMA20 ----
    df["signal_ema20"] = _compare_signal(close, df["ema_20"])

    # ---- RSI < 30 / exit > 70 ----
    rsi = df["rsi_14"]
    df["signal_rsi_lt30_gt70"] = _state_signal(rsi < 30, rsi > 70)

    # ---- RSI_avg crossover 50 ----
    rsi_avg = df["rsi_avg_14"]
    df["signal_rsi_avg_x50"] = _compare_signal(rsi_avg, pd.Series(50.0, index=df.index))

    # ---- RSI cross its own average ----
    df["signal_rsi_x_avg"] = _compare_signal(rsi, rsi_avg)

    # ---- RSI Bollinger Bands: enter below lower, exit above upper ----
    df["signal_rsi_bb"] = _state_signal(rsi < df["rsi_bb_lower"], rsi > df["rsi_bb_upper"])

    # ---- Bollinger mid (price vs mid) ----
    df["signal_bb_mid"] = _compare_signal(close, df["bb_mid"])

    # ---- BB lower/upper bounce: enter ≤ lower, exit ≥ upper ----
    df["signal_bb_lower_upper"] = _state_signal(
        close <= df["bb_lower"], close >= df["bb_upper"]
    )

    # ---- MFI ≤ 20 / exit ≥ 80 ----
    mfi = df["mfi_14"]
    df["signal_mfi"] = _state_signal(mfi <= 20, mfi >= 80)

    # ---- Stochastic %K level: enter < 20, exit > 80 ----
    sk = df["stoch_k"]
    df["signal_stoch_k_lvl"] = _state_signal(sk < 20, sk > 80)

    # ---- Stochastic %K / %D crossover ----
    # Standard momentum: long when %K > %D (upward cross)
    df["signal_stoch_kd_x"] = _compare_signal(sk, df["stoch_d"])

    # ---- Aroon Up/Down crossover ----
    df["signal_aroon_x"] = _compare_signal(df["aroon_up"], df["aroon_down"])

    # ---- Aroon filtered (aroon_up > 70 when entering, aroon_down > 70 when exiting) ----
    up = df["aroon_up"]
    down = df["aroon_down"]
    aroon_enter = (up > 70) & (up > down) & (up.shift(1) <= down.shift(1))
    aroon_exit = (down > 70) & (down > up) & (down.shift(1) <= up.shift(1))
    df["signal_aroon_filtered"] = _state_signal(aroon_enter, aroon_exit)

    # ---- MACD cross signal line ----
    df["signal_macd_x"] = _compare_signal(df["macd"], df["macd_signal"])

    # ---- MACD above zero ----
    df["signal_macd_zero"] = _compare_signal(df["macd"], pd.Series(0.0, index=df.index))

    # ---- VWAP ----
    df["signal_vwap"] = _compare_signal(close, df["vwap_daily_cum"])

    return df


# ---------------------------------------------------------------------------
# 6. Compute returns (no lookahead)
# ---------------------------------------------------------------------------

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each signal, compute:
        ret_xxx    : daily simple return (position_{t-1} * return_t)
        cumret_xxx : (1 + ret_xxx).cumprod()

    Uses shift(1) so today's signal drives tomorrow's return.
    """
    simple_ret = df["close"].pct_change()

    for name in SIGNAL_NAMES:
        sig_col = f"signal_{name}"
        ret_col = f"ret_{name}"
        cum_col = f"cumret_{name}"

        if sig_col not in df.columns:
            continue

        # Position held during bar t = signal at close of bar t-1
        position = df[sig_col].shift(1).fillna(0)
        df[ret_col] = position * simple_ret
        df[cum_col] = (1.0 + df[ret_col]).cumprod()

    return df


# ---------------------------------------------------------------------------
# 7. Summary statistics
# ---------------------------------------------------------------------------

def _max_drawdown(cumret: pd.Series) -> float:
    """Peak-to-trough maximum drawdown of a cumulative return series."""
    roll_max = cumret.cummax()
    drawdown = (cumret - roll_max) / roll_max
    return float(drawdown.min())


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary DataFrame (one row per signal rule) with columns:
        total_return, ann_return, ann_vol, sharpe, max_drawdown,
        win_rate, n_trades
    """
    rows = []
    n_bars = len(df)

    for name in SIGNAL_NAMES:
        sig_col = f"signal_{name}"
        ret_col = f"ret_{name}"
        cum_col = f"cumret_{name}"

        if sig_col not in df.columns or ret_col not in df.columns:
            continue

        rets = df[ret_col].dropna()
        cumret = df[cum_col].dropna()

        if cumret.empty:
            continue

        total_ret = float(cumret.iloc[-1]) - 1.0

        # Annualised return (CAGR)
        n_days = n_bars
        years = n_days / TRADING_DAYS_PER_YEAR
        ann_ret = (1.0 + total_ret) ** (1.0 / max(years, 1e-6)) - 1.0

        # Annualised volatility
        ann_vol = float(rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        # Sharpe (rf = 0)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

        # Max drawdown
        mdd = _max_drawdown(cumret)

        # Trade statistics
        sig = df[sig_col]
        entries = ((sig == 1) & (sig.shift(1) == 0)).sum()
        n_trades = int(entries)

        # Win rate: fraction of trades that ended with positive return
        # Identify trade start/end indices
        in_trade = sig.values
        trade_returns = []
        trade_start = None
        for i in range(len(in_trade)):
            if in_trade[i] == 1 and (i == 0 or in_trade[i - 1] == 0):
                trade_start = i
            elif in_trade[i] == 0 and i > 0 and in_trade[i - 1] == 1:
                if trade_start is not None:
                    # Return over trade window (using next-day returns already baked in)
                    trade_ret = rets.iloc[trade_start : i].sum()
                    trade_returns.append(trade_ret)
                    trade_start = None

        win_rate = (
            float(sum(r > 0 for r in trade_returns) / len(trade_returns))
            if trade_returns
            else np.nan
        )

        rows.append(
            {
                "signal": name,
                "total_return": round(total_ret, 4),
                "ann_return": round(ann_ret, 4),
                "ann_vol": round(ann_vol, 4),
                "sharpe": round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
                "max_drawdown": round(mdd, 4),
                "win_rate": round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
                "n_trades": n_trades,
            }
        )

    summary_df = pd.DataFrame(rows).set_index("signal")
    return summary_df


# ---------------------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------------------

def build_signals_and_returns(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1day",
    api_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: fetch → indicators → signals → returns → summary.

    Parameters
    ----------
    symbol   : ticker, e.g. 'AAPL'
    start    : 'YYYY-MM-DD'
    end      : 'YYYY-MM-DD'
    interval : '1day' or FMP intraday interval
    api_key  : overrides FMP_API_KEY env var

    Returns
    -------
    (df, summary_df)
        df         : full aligned DataFrame with all indicator, signal,
                     ret_*, and cumret_* columns
        summary_df : per-signal performance statistics
    """
    df = fetch_fmp_ohlcv(symbol, start, end, interval, api_key)
    df = _calc_all_indicators(df, interval)
    df = generate_all_signals(df)
    df = compute_returns(df)
    summary_df = compute_summary(df)
    return df, summary_df


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # Set your key: os.environ['FMP_API_KEY'] = 'YOUR_KEY'
    # Or pass api_key= directly to build_signals_and_returns

    df, summary = build_signals_and_returns("AAPL", "2022-01-01", "2024-01-01")

    print("\n=== Signal Summary ===")
    print(summary[["total_return", "sharpe", "max_drawdown", "n_trades"]].to_string())

    print("\n=== Last 10 bars (SMA20 signal) ===")
    print(
        df[["close", "sma_20", "signal_sma20", "ret_sma20", "cumret_sma20"]]
        .tail(10)
        .to_string()
    )

    # Basic sanity checks
    warm_up = 30
    for name in SIGNAL_NAMES:
        col = f"signal_{name}"
        if col in df.columns:
            assert df[col].iloc[warm_up:].notna().all(), f"NaN found in {col} after warm-up"

    print("\nAll signal columns pass NaN check after warm-up period.")
