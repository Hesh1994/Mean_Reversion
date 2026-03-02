"""
data_ingestion.py
=================
FMP OHLCV data ingestion module.

Features
--------
- Single-symbol and batch OHLCV fetching (daily + intraday)
- Automatic retry with exponential back-off (3 attempts)
- Optional CSV disk cache keyed by symbol / interval / date range
- Data cleaning: zero-volume → NaN, dtype enforcement, oldest-first sort
- Thin FMPClient class (reuses an HTTP session; holds API key)

Quick start
-----------
    import os
    os.environ["FMP_API_KEY"] = "YOUR_KEY"

    from data_ingestion import fetch_ohlcv, fetch_ohlcv_batch

    # Single symbol – daily
    df = fetch_ohlcv("AAPL", "2022-01-01", "2024-01-01")

    # Single symbol – intraday (5-minute bars)
    df_5m = fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31", interval="5min")

    # Multiple symbols
    frames = fetch_ohlcv_batch(["AAPL", "MSFT", "NVDA"], "2023-01-01", "2024-01-01")
    # frames["AAPL"] → pd.DataFrame

    # With on-disk caching (skips network if CSV already exists)
    df = fetch_ohlcv("AAPL", "2022-01-01", "2024-01-01", cache_dir="./cache")
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FMP_BASE = "https://financialmodelingprep.com/api/v3"

DAILY_INTERVAL = "1day"
INTRADAY_INTERVALS = {"1min", "5min", "15min", "30min", "1hour", "4hour"}
ALL_INTERVALS = {DAILY_INTERVAL} | INTRADAY_INTERVALS

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF = 1.5   # seconds; doubles each attempt
_REQUEST_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FMPError(Exception):
    """Raised when the FMP API returns an error or empty payload."""


class FMPAuthError(FMPError):
    """Raised when the API key is missing or rejected."""


# ---------------------------------------------------------------------------
# FMPClient
# ---------------------------------------------------------------------------

class FMPClient:
    """
    Thin wrapper around the FMP REST API.

    Parameters
    ----------
    api_key  : FMP API key. Falls back to ``FMP_API_KEY`` env var.
    base_url : Override the API base URL (useful for testing).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = FMP_BASE,
    ) -> None:
        self._key = api_key or os.environ.get("FMP_API_KEY", "")
        if not self._key:
            raise FMPAuthError(
                "FMP API key not found. Set the FMP_API_KEY environment variable "
                "or pass api_key= to FMPClient."
            )
        self._base = base_url.rstrip("/")
        self._session = self._build_session()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = DAILY_INTERVAL,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for one symbol.

        Parameters
        ----------
        symbol   : Ticker (case-insensitive), e.g. ``"AAPL"``.
        start    : Inclusive start date ``"YYYY-MM-DD"``.
        end      : Inclusive end date  ``"YYYY-MM-DD"``.
        interval : ``"1day"`` or any FMP intraday interval.

        Returns
        -------
        pd.DataFrame
            Columns: open, high, low, close, volume.
            DatetimeIndex sorted oldest-first.

        Raises
        ------
        FMPAuthError  : invalid / missing key.
        FMPError      : empty payload or unexpected response shape.
        requests.HTTPError : non-2xx HTTP status after retries.
        """
        _validate_interval(interval)
        symbol = symbol.upper()

        raw = self._get_with_retry(*self._build_request(symbol, start, end, interval))
        df = _parse_response(raw, interval)
        return _clean_ohlcv(df)

    def fetch_ohlcv_batch(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str = DAILY_INTERVAL,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV bars for multiple symbols sequentially.

        Returns
        -------
        dict mapping ticker → pd.DataFrame.
        Failed tickers are logged as warnings and omitted from the result.
        """
        results: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                results[sym.upper()] = self.fetch_ohlcv(sym, start, end, interval)
            except Exception as exc:
                log.warning("Could not fetch %s: %s", sym, exc)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_request(
        self, symbol: str, start: str, end: str, interval: str
    ):
        """Return (url, params) tuple for the correct FMP endpoint."""
        params = {"from": start, "to": end, "apikey": self._key}
        if interval == DAILY_INTERVAL:
            url = f"{self._base}/historical-price-full/{symbol}"
        else:
            url = f"{self._base}/historical-chart/{interval}/{symbol}"
        return url, params

    def _get_with_retry(self, url: str, params: dict) -> object:
        """
        GET with up to _RETRY_ATTEMPTS tries.
        Sleeps _RETRY_BACKOFF * 2^attempt seconds between retries.
        Returns parsed JSON payload.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = self._session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
                if resp.status_code == 401:
                    raise FMPAuthError("FMP returned 401 – check your API key.")
                resp.raise_for_status()
                return resp.json()
            except FMPAuthError:
                raise
            except Exception as exc:
                last_exc = exc
                sleep_s = _RETRY_BACKOFF * (2 ** attempt)
                log.warning(
                    "FMP request failed (attempt %d/%d): %s. Retrying in %.1fs…",
                    attempt + 1,
                    _RETRY_ATTEMPTS,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        # Mount a transport-level retry adapter (handles transient TCP errors)
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=0,  # we handle retries ourselves
                raise_on_status=False,
            )
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session


# ---------------------------------------------------------------------------
# Data parsing & cleaning (pure functions — no network I/O)
# ---------------------------------------------------------------------------

def _parse_response(payload: object, interval: str) -> pd.DataFrame:
    """
    Convert raw FMP JSON to a DataFrame before cleaning.

    Daily endpoint wraps data in {"historical": [...]}.
    Intraday endpoint returns a bare list.
    """
    if interval == DAILY_INTERVAL:
        if not isinstance(payload, dict):
            raise FMPError(f"Unexpected daily payload type: {type(payload)}")
        records = payload.get("historical", [])
    else:
        if not isinstance(payload, list):
            raise FMPError(f"Unexpected intraday payload type: {type(payload)}")
        records = payload

    if not records:
        raise FMPError("FMP returned an empty dataset for the requested symbol / range.")

    return pd.DataFrame(records)


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise a raw FMP DataFrame:
    - Parse ``date`` column → DatetimeIndex
    - Keep only OHLCV columns (those present)
    - Cast all columns to float64
    - Replace zero volume with NaN
    - Sort index oldest-first, drop exact duplicates
    """
    if "date" not in df.columns:
        raise FMPError("Response missing 'date' column.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    present = [c for c in OHLCV_COLS if c in df.columns]
    missing = set(OHLCV_COLS) - set(present)
    if missing:
        log.warning("Missing expected OHLCV columns: %s", missing)

    df = df[present].copy()
    df = df.astype(float)

    if "volume" in df.columns:
        df["volume"] = df["volume"].replace(0.0, np.nan)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, symbol: str, interval: str, start: str, end: str) -> Path:
    fname = f"{symbol}_{interval}_{start}_{end}.csv"
    return cache_dir / fname


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        log.info("Cache hit: %s", path.name)
        return df
    except Exception as exc:
        log.warning("Cache read failed (%s): %s – fetching fresh.", path.name, exc)
        return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    except Exception as exc:
        log.warning("Cache write failed (%s): %s", path.name, exc)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol: str,
    start: str,
    end: str,
    interval: str = DAILY_INTERVAL,
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single symbol from FMP.

    Parameters
    ----------
    symbol    : Ticker, e.g. ``"AAPL"``.
    start     : ``"YYYY-MM-DD"``
    end       : ``"YYYY-MM-DD"``
    interval  : ``"1day"`` (default) or intraday interval.
    api_key   : Overrides ``FMP_API_KEY`` env var.
    cache_dir : Directory for CSV caching. Pass ``None`` to disable.

    Returns
    -------
    pd.DataFrame – open, high, low, close, volume; DatetimeIndex oldest-first.
    """
    symbol = symbol.upper()

    if cache_dir is not None:
        cache_path = _cache_path(Path(cache_dir), symbol, interval, start, end)
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    client = FMPClient(api_key=api_key)
    df = client.fetch_ohlcv(symbol, start, end, interval)

    if cache_dir is not None:
        _save_cache(df, cache_path)

    return df


def fetch_ohlcv_batch(
    symbols: List[str],
    start: str,
    end: str,
    interval: str = DAILY_INTERVAL,
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple symbols.

    Parameters
    ----------
    symbols   : List of tickers.
    start     : ``"YYYY-MM-DD"``
    end       : ``"YYYY-MM-DD"``
    interval  : ``"1day"`` or intraday interval.
    api_key   : Overrides ``FMP_API_KEY`` env var.
    cache_dir : Directory for CSV caching. Pass ``None`` to disable.

    Returns
    -------
    dict mapping ticker → pd.DataFrame.
    Failed tickers are logged and omitted.
    """
    results: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            results[sym.upper()] = fetch_ohlcv(sym, start, end, interval, api_key, cache_dir)
        except Exception as exc:
            log.warning("Skipping %s: %s", sym, exc)
    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_interval(interval: str) -> None:
    if interval not in ALL_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. "
            f"Choose from: {sorted(ALL_INTERVALS)}"
        )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    start = sys.argv[2] if len(sys.argv) > 2 else "2023-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"

    print(f"Fetching {sym} daily from {start} to {end} …")
    df = fetch_ohlcv(sym, start, end, cache_dir="./fmp_cache")
    print(df.tail(5).to_string())
    print(f"\nShape: {df.shape}  |  NaN volume rows: {df['volume'].isna().sum()}")
