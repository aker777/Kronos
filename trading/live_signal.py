"""Generate today's BUY / HOLD / SELL signals for the configured universe.

Fetches the latest bars per ticker, forecasts the next `pred_len` bars with
Kronos, and prints a signal table. You place any trades manually.

Usage:
    python -m trading.live_signal                    # tradable universe from config
    python -m trading.live_signal --all              # include research-only tickers
    python -m trading.live_signal --tickers CW8.PA AAPL
    python -m trading.live_signal --log              # append to trading/signals_log.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import os

import pandas as pd

from trading.config import load_config, iter_universe
from trading.data import load_ticker
from trading.predictor import KronosSignalModel
from trading.signals import signal_from_dist


def main() -> None:
    ap = argparse.ArgumentParser(description="Kronos live signals")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--all", action="store_true", help="include research-only tickers")
    ap.add_argument("--log", action="store_true", help="append results to signals_log.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sig_cfg = cfg["signal"]
    if args.tickers:
        entries = [(None, t, t, True) for t in args.tickers]
    else:
        entries = list(iter_universe(cfg, tradable_only=not args.all))

    model = KronosSignalModel(cfg)
    print(f"Loaded {cfg['model']['name']} on {model.device}. "
          f"Horizon: {cfg['data']['pred_len']} x {cfg['data']['interval']}.\n")

    rows = []
    for _region, ticker, name, tradable in entries:
        try:
            df = load_ticker(ticker, cfg, refresh=True)  # want the freshest bars
            dist = model.forecast_dist(df, end_idx=None,
                                       n_samples=sig_cfg.get("sample_count", 10),
                                       T=sig_cfg["T"], top_p=sig_cfg["top_p"])
            sig = signal_from_dist(dist, sig_cfg)
            rows.append({
                "ticker": ticker,
                "name": name,
                "tradable": tradable,
                "last_close": round(sig.last_close, 4),
                "pred_return": round(sig.horizon_return, 4),
                "signal": sig.action,
                "confidence": round(sig.confidence, 2),
                "as_of": df["timestamps"].iloc[-1].date().isoformat(),
            })
        except Exception as e:
            rows.append({"ticker": ticker, "name": name, "signal": "ERROR",
                         "pred_return": None, "confidence": None, "last_close": None,
                         "tradable": tradable, "as_of": str(e)[:60]})

    table = pd.DataFrame(rows)
    order = {"BUY": 0, "SELL": 1, "HOLD": 2, "ERROR": 3}
    table = table.sort_values(by="signal", key=lambda s: s.map(order).fillna(9))
    print(table.to_string(index=False))

    if args.log:
        table.insert(0, "generated_at", dt.datetime.now().isoformat(timespec="seconds"))
        log_path = os.path.join(os.path.dirname(__file__), "signals_log.csv")
        table.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
        print(f"\nAppended to {log_path}")


if __name__ == "__main__":
    main()
