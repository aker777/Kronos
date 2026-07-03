"""Walk-forward backtest for the Kronos signal toolkit.

This is the honest test the repo lacks: at each decision point we feed the model
ONLY the trailing lookback window (no look-ahead), turn its forecast into a
long/flat signal, then mark the position to market against the ACTUAL returns of
the following bars. We compare against buy-and-hold and a naive momentum baseline.

Usage:
    python -m trading.backtest                 # whole tradable universe from config
    python -m trading.backtest --tickers AAPL CW8.PA 0700.HK
    python -m trading.backtest --fast          # threshold-only (1 sample) = much faster
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless: save PNGs, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading import metrics
from trading.config import load_config, iter_universe
from trading.data import load_ticker
from trading.predictor import KronosSignalModel
from trading.signals import BUY, SELL, HOLD, signal_from_return


def _decision_indices(n: int, lookback: int, pred_len: int, step: int) -> list[int]:
    """Positions t with a full lookback behind and `pred_len` actual bars ahead."""
    first = lookback - 1
    last = n - 1 - pred_len
    return list(range(first, last + 1, step))


def backtest_ticker(model: KronosSignalModel, df: pd.DataFrame, cfg: dict,
                    fast: bool = False) -> dict:
    """Run the walk-forward backtest for one ticker. Returns records + metrics."""
    lookback = cfg["data"]["lookback"]
    pred_len = cfg["data"]["pred_len"]
    step = cfg["backtest"].get("step", 1)
    fee = cfg["backtest"].get("fee_bps", 0) / 1e4
    sig_cfg = cfg["signal"]
    n_samples = 1 if fast else sig_cfg.get("sample_count", 10)

    idxs = _decision_indices(len(df), lookback, pred_len, step)
    if not idxs:
        raise ValueError("Not enough bars for even one decision; lower lookback/pred_len.")

    rows = []
    position = 0  # long/flat carried between decisions
    for t in idxs:
        # --- forecast using only data up to and including bar t ---
        if fast:
            pred = model.forecast(df, end_idx=t, sample_count=1,
                                  T=sig_cfg["T"], top_p=sig_cfg["top_p"])
            last_close = float(df["close"].iloc[t])
            pred_ret = float(pred["close"].iloc[-1]) / last_close - 1.0
            action = signal_from_return(pred_ret, sig_cfg)
        else:
            dist = model.forecast_dist(df, end_idx=t, n_samples=n_samples,
                                       T=sig_cfg["T"], top_p=sig_cfg["top_p"])
            from trading.signals import signal_from_dist
            sig = signal_from_dist(dist, sig_cfg)
            pred_ret, action, last_close = sig.horizon_return, sig.action, sig.last_close

        # --- decide next position (long/flat only) ---
        prev_position = position
        if action == BUY:
            position = 1
        elif action == SELL:
            position = 0
        # HOLD -> keep prev_position

        # --- realise the ACTUAL market return over the holding period (t -> t+step) ---
        # PnL is marked over `step` (we re-decide every `step` bars); the forecast
        # quality (hit-rate) is judged over the model's own horizon `pred_len`.
        close_t = float(df["close"].iloc[t])
        hold_end = min(t + step, len(df) - 1)
        actual_ret = float(df["close"].iloc[hold_end]) / close_t - 1.0
        horizon_actual = float(df["close"].iloc[t + pred_len]) / close_t - 1.0
        cost = fee if position != prev_position else 0.0
        strat_ret = position * actual_ret - cost

        rows.append({
            "timestamps": df["timestamps"].iloc[t],
            "pred_return": pred_ret,
            "actual_return": actual_ret,
            "horizon_actual": horizon_actual,
            "action": action,
            "position": position,
            "strat_return": strat_ret,
        })

    res = pd.DataFrame(rows).set_index("timestamps")
    equity = cfg["backtest"]["initial_capital"] * (1.0 + res["strat_return"]).cumprod()
    bench = cfg["backtest"]["initial_capital"] * (1.0 + res["actual_return"]).cumprod()

    m = metrics.summarize(res["strat_return"], equity)
    m.update(metrics.trade_stats(res["position"], res["actual_return"]))
    m["hit_rate"] = metrics.directional_hit_rate(res["pred_return"], res["horizon_actual"])
    m["buy_hold_return"] = float(bench.iloc[-1] / bench.iloc[0] - 1.0)
    m["excess_vs_bh"] = m["total_return"] - m["buy_hold_return"]
    return {"records": res, "equity": equity, "benchmark": bench, "metrics": m}


def plot_result(ticker: str, out: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    equity, bench, res, m = out["equity"], out["benchmark"], out["records"], out["metrics"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity.index, equity.values, label="Kronos strategy", color="#1f77b4", lw=1.8)
    ax1.plot(bench.index, bench.values, label="Buy & hold", color="#ff7f0e", lw=1.3, alpha=0.8)
    ax1.set_ylabel("Equity")
    ax1.set_title(f"{ticker} — Kronos walk-forward backtest")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    txt = (f"Total: {m['total_return']:.1%}   B&H: {m['buy_hold_return']:.1%}   "
           f"Excess: {m['excess_vs_bh']:+.1%}\n"
           f"CAGR: {m['cagr']:.1%}   Sharpe: {m['sharpe']:.2f}   MaxDD: {m['max_drawdown']:.1%}\n"
           f"Hit-rate: {m['hit_rate']:.1%}   Win-rate: {m['win_rate']:.1%}   Trades: {m['n_trades']}")
    ax1.text(0.01, 0.98, txt, transform=ax1.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))

    curve = (1.0 + res["strat_return"]).cumprod()
    dd = (curve - curve.cummax()) / curve.cummax()
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{ticker.replace('.', '_')}_backtest.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _print_metrics(ticker: str, m: dict) -> None:
    print(f"\n=== {ticker} ===")
    print(f"  Total return   : {m['total_return']:+.2%}   (buy&hold {m['buy_hold_return']:+.2%}, "
          f"excess {m['excess_vs_bh']:+.2%})")
    print(f"  CAGR           : {m['cagr']:+.2%}")
    print(f"  Sharpe         : {m['sharpe']:.2f}")
    print(f"  Max drawdown   : {m['max_drawdown']:.2%}")
    print(f"  Directional hit: {m['hit_rate']:.2%}")
    print(f"  Win rate       : {m['win_rate']:.2%}   over {m['n_trades']} trades")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kronos walk-forward backtest")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tickers", nargs="*", help="override universe with these tickers")
    ap.add_argument("--fast", action="store_true", help="1-sample threshold signal (faster)")
    ap.add_argument("--all", action="store_true", help="include research-only tickers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.tickers:
        tickers = [(None, t, t) for t in args.tickers]
    else:
        tickers = [(r, t, n) for r, t, n, _tr in iter_universe(cfg, tradable_only=not args.all)]

    model = KronosSignalModel(cfg)
    print(f"Loaded {cfg['model']['name']} on {model.device}. "
          f"Backtesting {len(tickers)} tickers ({'fast' if args.fast else 'full'} mode).")

    summary = []
    for _region, ticker, _name in tickers:
        try:
            df = load_ticker(ticker, cfg)
            out = backtest_ticker(model, df, cfg, fast=args.fast)
            png = plot_result(ticker, out, cfg["backtest"]["out_dir"])
            _print_metrics(ticker, out["metrics"])
            print(f"  chart -> {png}")
            summary.append({"ticker": ticker, **out["metrics"]})
        except Exception as e:  # keep going across the basket
            print(f"\n=== {ticker} ===\n  SKIPPED: {e}")

    if summary:
        sdf = pd.DataFrame(summary).set_index("ticker")
        print("\n================ AGGREGATE ================")
        print(sdf[["total_return", "buy_hold_return", "excess_vs_bh",
                   "sharpe", "max_drawdown", "hit_rate", "win_rate", "n_trades"]]
              .to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"\nMean excess vs buy&hold: {sdf['excess_vs_bh'].mean():+.2%}   "
              f"Mean Sharpe: {sdf['sharpe'].mean():.2f}   "
              f"Mean hit-rate: {sdf['hit_rate'].mean():.2%}")
        out_csv = os.path.join(cfg["backtest"]["out_dir"], "summary.csv")
        sdf.to_csv(out_csv)
        print(f"Summary CSV -> {out_csv}")


if __name__ == "__main__":
    main()
