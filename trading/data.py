"""Global OHLCV data layer for the Kronos signal toolkit.

Fetches candlesticks with yfinance (developed + emerging markets: Europe, US,
Asia, Africa, LatAm) and normalises them to the exact schema Kronos expects:

    timestamps, open, high, low, close, volume, amount

`amount` (turnover) is not provided by yfinance, so it is set to 0 — Kronos
tolerates a zero amount column. The layer is generic over `interval` ("1d" now,
"1h" later) so no downstream code changes when you switch timeframe.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

KRONOS_COLS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


def _cache_path(cache_dir: str, ticker: str, interval: str) -> str:
    safe = ticker.replace("/", "_").replace(".", "_")
    return os.path.join(cache_dir, f"{safe}_{interval}.csv")


def fetch_ohlcv(
    ticker: str,
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    cache_dir: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV for `ticker` and return a Kronos-schema DataFrame.

    Columns: timestamps, open, high, low, close, volume, amount — ascending by
    time, no NaNs, timezone-naive timestamps.

    Set `cache_dir` to read/write a CSV cache (skips the network on repeat runs
    unless `refresh=True`).
    """
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = _cache_path(cache_dir, ticker, interval)
        if cache_file and os.path.exists(cache_file) and not refresh:
            df = pd.read_csv(cache_file, parse_dates=["timestamps"])
            return df[KRONOS_COLS]

    import yfinance as yf  # imported lazily so the module loads without yfinance

    raw = yf.download(
        ticker,
        interval=interval,
        start=start,
        end=end,
        auto_adjust=True,      # adjust for splits/dividends -> cleaner returns
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker} (interval={interval}).")

    # yfinance may return MultiIndex columns for a single ticker; flatten.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = raw.reset_index().rename(columns={"Date": "timestamps", "Datetime": "timestamps"})

    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: missing expected columns {missing}.")

    df["amount"] = 0.0
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
    df = df[KRONOS_COLS].dropna().sort_values("timestamps").reset_index(drop=True)

    if cache_file:
        df.to_csv(cache_file, index=False)
    return df


def check_quality(
    df: pd.DataFrame,
    ticker: str,
    interval: str = "1d",
    min_bars: int = 250,
    max_gap_days: int = 10,
) -> list[str]:
    """Run data-quality guards (important for thin emerging-market names).

    Returns a list of human-readable warning strings; empty list means clean.
    Raises ValueError if the series is too short to use at all.
    """
    warns: list[str] = []
    if len(df) < min_bars:
        raise ValueError(
            f"{ticker}: only {len(df)} bars (< min_bars={min_bars}); too short to use."
        )

    # Long calendar gaps (holidays, halts, illiquidity) — daily only.
    if interval == "1d" and len(df) > 1:
        gaps = df["timestamps"].diff().dt.days.dropna()
        big = int((gaps > max_gap_days).sum())
        if big:
            warns.append(f"{ticker}: {big} gaps larger than {max_gap_days} days.")

    # Zero-volume stretches (no trading / bad data).
    zero_vol = int((df["volume"] <= 0).sum())
    if zero_vol:
        warns.append(f"{ticker}: {zero_vol} zero-volume bars ({zero_vol/len(df):.1%}).")

    # Stale closes (repeated identical prints -> illiquid / stale feed).
    stale = int((df["close"].diff() == 0).sum())
    if stale > 0.10 * len(df):
        warns.append(f"{ticker}: {stale} stale (unchanged) closes ({stale/len(df):.1%}).")

    for w in warns:
        warnings.warn(w, stacklevel=2)
    return warns


def load_ticker(ticker: str, cfg: dict, refresh: bool = False) -> pd.DataFrame:
    """Convenience: fetch + quality-check a ticker using the config's data settings."""
    d = cfg["data"]
    df = fetch_ohlcv(
        ticker,
        interval=d["interval"],
        start=d.get("history_start"),
        cache_dir=d.get("cache_dir"),
        refresh=refresh,
    )
    check_quality(
        df,
        ticker,
        interval=d["interval"],
        min_bars=d.get("min_bars", 250),
        max_gap_days=d.get("max_gap_days", 10),
    )
    return df
