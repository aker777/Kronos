"""Thin wrapper around Kronos: load the model once, forecast a lookback window.

`predict()` in the core model averages `sample_count` Monte-Carlo samples into a
single point forecast (model/kronos.py:467). For a directional-*vote* confidence
we also expose `forecast_dist()`, which runs several single-sample forecasts and
returns the distribution of horizon returns.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Make the repo root importable so `from model import ...` works from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

_PRICE_VOL = ["open", "high", "low", "close", "volume", "amount"]


def make_future_timestamps(last_ts: pd.Timestamp, pred_len: int, interval: str) -> pd.Series:
    """Generate `pred_len` future timestamps after `last_ts` for the interval.

    Daily -> business days; hourly -> hourly steps. Only used for the model's
    calendar features, so approximate spacing is fine.
    """
    last_ts = pd.Timestamp(last_ts)
    if interval.endswith("d"):
        idx = pd.bdate_range(start=last_ts, periods=pred_len + 1, freq="B")[1:]
    elif interval.endswith("h"):
        step = int(interval[:-1] or 1)
        idx = pd.date_range(start=last_ts, periods=pred_len + 1, freq=f"{step}h")[1:]
    else:
        idx = pd.date_range(start=last_ts, periods=pred_len + 1)[1:]
    return pd.Series(idx)


class KronosSignalModel:
    """Loads Kronos once; forecasts trailing windows of a Kronos-schema DataFrame."""

    def __init__(self, cfg: dict):
        m = cfg["model"]
        self.interval = cfg["data"]["interval"]
        self.lookback = cfg["data"]["lookback"]
        self.pred_len = cfg["data"]["pred_len"]

        tokenizer = KronosTokenizer.from_pretrained(m["tokenizer"])
        model = Kronos.from_pretrained(m["name"])
        self.predictor = KronosPredictor(
            model, tokenizer, device=m.get("device"), max_context=m["max_context"]
        )
        self.device = self.predictor.device

    def _window(self, df: pd.DataFrame, end_idx: int | None):
        """Return (x_df, x_ts, y_ts, last_close) for a lookback ending at end_idx.

        end_idx is the position of the last *observed* bar (inclusive). If None,
        uses the final row (live forecast).
        """
        if end_idx is None:
            end_idx = len(df) - 1
        start = max(0, end_idx - self.lookback + 1)
        window = df.iloc[start : end_idx + 1]
        x_df = window[_PRICE_VOL].reset_index(drop=True)
        x_ts = window["timestamps"].reset_index(drop=True)
        y_ts = make_future_timestamps(x_ts.iloc[-1], self.pred_len, self.interval)
        last_close = float(window["close"].iloc[-1])
        return x_df, x_ts, y_ts, last_close

    def forecast(
        self,
        df: pd.DataFrame,
        end_idx: int | None = None,
        sample_count: int = 10,
        T: float = 1.0,
        top_p: float = 0.9,
    ) -> pd.DataFrame:
        """Averaged point forecast (OHLCV) for the horizon after `end_idx`."""
        x_df, x_ts, y_ts, _ = self._window(df, end_idx)
        return self.predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=self.pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )

    def forecast_dist(
        self,
        df: pd.DataFrame,
        end_idx: int | None = None,
        n_samples: int = 10,
        T: float = 1.0,
        top_p: float = 0.9,
    ) -> dict:
        """Distribution of horizon outcomes from `n_samples` single-sample rollouts.

        Returns a dict with:
          last_close      : last observed close
          mean_close      : mean predicted close at the horizon end
          horizon_return  : mean_close / last_close - 1
          up_vote         : fraction of samples whose horizon return > 0
          sample_returns  : np.ndarray of per-sample horizon returns
        """
        x_df, x_ts, y_ts, last_close = self._window(df, end_idx)
        final_closes = np.empty(n_samples, dtype=float)
        for i in range(n_samples):
            pred = self.predictor.predict(
                df=x_df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=self.pred_len,
                T=T,
                top_p=top_p,
                sample_count=1,
                verbose=False,
            )
            final_closes[i] = float(pred["close"].iloc[-1])
        sample_returns = final_closes / last_close - 1.0
        mean_close = float(final_closes.mean())
        return {
            "last_close": last_close,
            "mean_close": mean_close,
            "horizon_return": mean_close / last_close - 1.0,
            "up_vote": float((sample_returns > 0).mean()),
            "sample_returns": sample_returns,
        }
