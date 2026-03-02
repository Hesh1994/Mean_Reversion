"""
mean_reversion.py
=================
Mean reversion strategy engine.

Provides z-score computation, Ornstein-Uhlenbeck half-life estimation,
state-machine signal generation, and rolling performance metrics.

Usage
-----
    from mean_reversion import build_mr_strategy, rolling_strategy_metrics

    df, stats = build_mr_strategy(ohlcv_df, zscore_window=20,
                                   entry_z=1.5, exit_z=0.5,
                                   allow_short=False, use_ema=False)

Design Notes
------------
- Z-score: (price − SMA) / rolling_std(ddof=1)  or  (price − EMA) / rolling_std
- EMA z-score uses pandas ewm(span=window, adjust=False) — NOT SMA-seeded
- Half-life: OLS on OU process ΔP = α + β·P_{t-1}; hl = −ln(2)/β
- Signals: no-lookahead (position_t applied to return_{t+1} via shift(1))
- Short signal is −1; long is +1; flat is 0
"""

import warnings
import numpy as np
import pandas as pd
from typing import Tuple

warnings.filterwarnings("ignore")

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# 1. Z-Score helpers
# ---------------------------------------------------------------------------

def calc_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    SMA-based z-score: (price − SMA(window)) / rolling_std(window, ddof=1).
    Zero or NaN std → NaN.
    """
    sma = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=1)
    std = std.replace(0.0, np.nan)
    return (series - sma) / std


def calc_zscore_ema(series: pd.Series, window: int) -> pd.Series:
    """
    EMA-based z-score: (price − EWM(span=window)) / rolling_std(window, ddof=1).
    Zero or NaN std → NaN.
    """
    ema = series.ewm(span=window, adjust=False).mean()
    std = series.rolling(window).std(ddof=1)
    std = std.replace(0.0, np.nan)
    return (series - ema) / std


# ---------------------------------------------------------------------------
# 2. Ornstein-Uhlenbeck half-life
# ---------------------------------------------------------------------------

def calc_half_life(series: pd.Series, min_obs: int = 20) -> float:
    """
    Estimate mean-reversion half-life via OLS on the OU process:
        ΔP_t = α + β·P_{t-1} + ε

    half_life = −ln(2) / β

    Returns np.nan when:
    - Fewer than min_obs valid observations
    - β ≥ 0  (series is not mean-reverting)
    - OLS fails
    """
    series = series.dropna()
    if len(series) < min_obs:
        return np.nan

    delta = series.diff().dropna()
    lag = series.shift(1).dropna()

    # Align on common index
    common = delta.index.intersection(lag.index)
    delta = delta.loc[common]
    lag = lag.loc[common]

    if len(delta) < min_obs:
        return np.nan

    # OLS: [1, lag] → delta
    X = np.column_stack([np.ones(len(lag)), lag.values])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, delta.values, rcond=None)
    except Exception:
        return np.nan

    beta = coeffs[1]
    if beta >= 0:
        return np.nan  # Not mean-reverting

    return float(-np.log(2) / beta)


def calc_rolling_half_life(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Slide calc_half_life over a rolling window.
    First window−1 values are NaN.
    Shows how mean-reversion speed changes over time.
    """
    n = len(series)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = series.iloc[i - window + 1 : i + 1]
        result[i] = calc_half_life(window_data)

    return pd.Series(result, index=series.index, name="rolling_hl")


# ---------------------------------------------------------------------------
# 3. Signal generation
# ---------------------------------------------------------------------------

def generate_mr_signals(
    df: pd.DataFrame,
    zscore_col: str,
    entry_z: float,
    exit_z: float,
    allow_short: bool,
) -> pd.DataFrame:
    """
    State-machine mean-reversion signal generator.

    Rules
    -----
    Long  : enter when z < −entry_z; exit when z > −exit_z
    Short : enter when z > +entry_z; exit when z < +exit_z  (if allow_short)
    NaN z-score → hold current state but output 0 for that bar.

    Output column ``signal_mr``: +1 (long), −1 (short), 0 (flat).
    """
    z = df[zscore_col].values
    n = len(z)
    signal = np.zeros(n, dtype=int)
    state = 0  # 0 = flat, 1 = long, -1 = short

    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            signal[i] = 0
            continue

        if state == 0:
            if zi < -entry_z:
                state = 1
            elif allow_short and zi > entry_z:
                state = -1
        elif state == 1:
            if zi > -exit_z:
                state = 0
        elif state == -1:
            if zi < exit_z:
                state = 0

        signal[i] = state

    df = df.copy()
    df["signal_mr"] = signal
    return df


# ---------------------------------------------------------------------------
# 4. Return computation
# ---------------------------------------------------------------------------

def compute_mr_returns(
    df: pd.DataFrame,
    signal_col: str = "signal_mr",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    No-lookahead return computation:
        position_t = signal_{t−1}   (shift(1))
        ret_mr     = position_t × price_return_t

    Adds columns: ret_mr, cumret_mr, bh_ret, bh_cumret.
    """
    df = df.copy()
    simple_ret = df[price_col].pct_change()
    position = df[signal_col].shift(1).fillna(0)

    df["ret_mr"] = position * simple_ret
    df["cumret_mr"] = (1.0 + df["ret_mr"]).cumprod()
    df["bh_ret"] = simple_ret
    df["bh_cumret"] = (1.0 + simple_ret).cumprod()
    return df


# ---------------------------------------------------------------------------
# 5. Summary statistics
# ---------------------------------------------------------------------------

def _max_drawdown(cumret: pd.Series) -> float:
    """Peak-to-trough maximum drawdown of a cumulative return series."""
    roll_max = cumret.cummax()
    drawdown = (cumret - roll_max) / roll_max
    return float(drawdown.min())


def _win_rate_from_trades(rets: pd.Series, signal: pd.Series) -> float:
    """
    Compute win rate: fraction of completed trades with positive total return.
    A trade is a contiguous run of non-zero signal.
    """
    sig = signal.values
    trade_returns = []
    trade_start = None

    for i in range(len(sig)):
        entered = sig[i] != 0
        prev_entered = sig[i - 1] != 0 if i > 0 else False

        if entered and not prev_entered:
            trade_start = i
        elif not entered and prev_entered and trade_start is not None:
            trade_ret = float(rets.iloc[trade_start:i].sum())
            trade_returns.append(trade_ret)
            trade_start = None

    if not trade_returns:
        return np.nan
    return float(sum(r > 0 for r in trade_returns) / len(trade_returns))


def compute_mr_stats(
    df: pd.DataFrame,
    signal_col: str = "signal_mr",
    ret_col: str = "ret_mr",
    cumret_col: str = "cumret_mr",
    bh_ret_col: str = "bh_ret",
    bh_cumret_col: str = "bh_cumret",
) -> pd.DataFrame:
    """
    CAGR, Sharpe, max_drawdown, win_rate, exposure for:
    - Mean reversion strategy
    - Buy-and-hold baseline

    Returns DataFrame indexed by strategy name.
    """
    rows = []
    n_bars = len(df)
    years = n_bars / TRADING_DAYS_PER_YEAR

    specs = [
        ("mean_reversion", ret_col, cumret_col, True),
        ("buy_and_hold",   bh_ret_col, bh_cumret_col, False),
    ]

    for label, rc, cc, is_mr in specs:
        if rc not in df.columns or cc not in df.columns:
            continue

        rets = df[rc].dropna()
        cumret = df[cc].dropna()

        if cumret.empty:
            continue

        total_ret = float(cumret.iloc[-1]) - 1.0
        ann_ret = (1.0 + total_ret) ** (1.0 / max(years, 1e-6)) - 1.0
        ann_vol = float(rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        mdd = _max_drawdown(cumret)
        exposure = float((df[signal_col] != 0).mean()) if is_mr else 1.0
        win_rate = _win_rate_from_trades(rets, df[signal_col]) if is_mr else np.nan

        rows.append({
            "strategy":     label,
            "total_return": round(total_ret, 4),
            "ann_return":   round(ann_ret,   4),
            "ann_vol":      round(ann_vol,   4),
            "sharpe":       round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
            "max_drawdown": round(mdd, 4),
            "win_rate":     round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
            "exposure":     round(exposure, 4),
        })

    return pd.DataFrame(rows).set_index("strategy")


# ---------------------------------------------------------------------------
# 6. Rolling strategy metrics  (used for alpha decay)
# ---------------------------------------------------------------------------

def rolling_strategy_metrics(
    df: pd.DataFrame,
    signal_col: str,
    ret_col: str,
    window: int = 20,
) -> pd.DataFrame:
    """
    Day-by-day rolling Sharpe, hit_rate, avg_ret, exposure over a rolling window.

    All metrics are annualised where appropriate.
    NaN is returned for windows with fewer than 2 observations.

    Returns DataFrame aligned with df.index.
    """
    rets = df[ret_col].fillna(0).values
    sig = df[signal_col].values
    n = len(df)

    rolling_sharpe = np.full(n, np.nan)
    hit_rate       = np.full(n, np.nan)
    avg_ret        = np.full(n, np.nan)
    exposure       = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        w_rets = rets[start : i + 1]
        w_sig  = sig[start  : i + 1]

        if len(w_rets) < 2:
            continue

        mean_r = float(np.mean(w_rets))
        std_r  = float(np.std(w_rets, ddof=1))

        rolling_sharpe[i] = (mean_r / std_r * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_r > 0 else np.nan
        hit_rate[i]       = float(np.mean(w_rets > 0))
        avg_ret[i]        = mean_r
        exposure[i]       = float(np.mean(w_sig != 0))

    return pd.DataFrame(
        {
            "rolling_sharpe": rolling_sharpe,
            "hit_rate":       hit_rate,
            "avg_ret":        avg_ret,
            "exposure":       exposure,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# 7. Full pipeline
# ---------------------------------------------------------------------------

def build_mr_strategy(
    df: pd.DataFrame,
    zscore_window: int = 20,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
    allow_short: bool = False,
    use_ema: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full mean-reversion pipeline:
        1. Compute z-score (SMA or EMA)
        2. Compute rolling half-life (window = 60)
        3. Generate state-machine signals
        4. Compute strategy and buy-and-hold returns
        5. Compute summary statistics

    Returns
    -------
    (df, stats_df)
        df       : input DataFrame enriched with zscore, rolling_hl,
                   signal_mr, ret_mr, cumret_mr, bh_ret, bh_cumret
        stats_df : per-strategy summary (mean_reversion, buy_and_hold)
    """
    df = df.copy()

    # Z-score
    if use_ema:
        df["zscore"] = calc_zscore_ema(df["close"], zscore_window)
    else:
        df["zscore"] = calc_zscore(df["close"], zscore_window)

    # Rolling half-life (fixed 60-bar window — independent of zscore_window)
    df["rolling_hl"] = calc_rolling_half_life(df["close"], window=60)

    # Signals
    df = generate_mr_signals(df, "zscore", entry_z, exit_z, allow_short)

    # Returns
    df = compute_mr_returns(df)

    # Stats
    stats_df = compute_mr_stats(df)

    return df, stats_df
