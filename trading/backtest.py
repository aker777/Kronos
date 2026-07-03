"""Walk-forward backtest for the Kronos signal toolkit.

This is the honest test the repo lacks: at each decision point we feed the model
ONLY the trailing lookback window (no look-ahead), turn its forecast into a
long/flat signal, then mark the position to market against the ACTUAL returns of
the following bars. We compare against buy-and-hold and a naive momentum baseline.

Usage:
    python -m trading.backtest                 # whole tradable universe from config
    python -m trading.backtest --tickers AAPL CW8.PA 0700.HK
    python -m trading.backtest --fast          # threshold-only (1 sample) = much faster
    python -m trading.backtest --end 2025-01-01                # tuning window only
    python -m trading.backtest --oos-start 2025-01-01          # separate OOS metrics
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless: save PNGs, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from trading import metrics
from trading.config import load_config, iter_universe
from trading.data import load_ticker
from trading.predictor import KronosSignalModel
from trading.signals import BUY, SELL, HOLD, signal_from_return, signal_from_dist


def _decision_indices(n: int, lookback: int, pred_len: int, step: int) -> list[int]:
    """Positions t with a full lookback behind and `pred_len` actual bars ahead."""
    first = lookback - 1
    last = n - 1 - pred_len
    return list(range(first, last + 1, step))


def simulate_positions(actions, actual_returns, fee: float):
    """Long/flat position path implied by a sequence of BUY/HOLD/SELL actions.

    Charges `fee` (fractional) whenever the position changes. Returns
    (positions, strat_returns) lists aligned with the inputs. Shared by the
    backtest and trading/sweep.py so the PnL rules can never drift apart.
    """
    position = 0
    positions: list[int] = []
    strat: list[float] = []
    for action, ret in zip(actions, actual_returns):
        prev = position
        if action == BUY:
            position = 1
        elif action == SELL:
            position = 0
        # HOLD -> keep previous position
        cost = fee if position != prev else 0.0
        positions.append(position)
        strat.append(position * float(ret) - cost)
    return positions, strat


def _compute_metrics(res: pd.DataFrame, capital: float):
    """Metrics + equity curves (strategy, buy&hold, momentum) for a records frame.

    All totals are measured from the same starting `capital` so the excesses
    are a fair comparison.
    """
    equity = capital * (1.0 + res["strat_return"]).cumprod()
    bench = capital * (1.0 + res["actual_return"]).cumprod()
    mom = capital * (1.0 + res["mom_return"]).cumprod()

    m = metrics.summarize(res["strat_return"], equity)
    m.update(metrics.trade_stats(res["position"], res["actual_return"]))
    m["hit_rate"] = metrics.directional_hit_rate(res["pred_return"], res["horizon_actual"])
    m["total_return"] = float(equity.iloc[-1] / capital - 1.0)
    if len(res) > 1:
        years = max((res.index[-1] - res.index[0]).days / 365.25, 1e-9)
        m["cagr"] = (1.0 + m["total_return"]) ** (1.0 / years) - 1.0 \
            if m["total_return"] > -1 else -1.0
    m["buy_hold_return"] = float(bench.iloc[-1] / capital - 1.0)
    m["momentum_return"] = float(mom.iloc[-1] / capital - 1.0)
    m["excess_vs_bh"] = m["total_return"] - m["buy_hold_return"]
    m["excess_vs_mom"] = m["total_return"] - m["momentum_return"]
    return m, equity, bench, mom


def backtest_ticker(model: KronosSignalModel, df: pd.DataFrame, cfg: dict,
                    fast: bool = False, limit: int | None = None,
                    ticker: str = "") -> dict:
    """Run the walk-forward backtest for one ticker. Returns records + metrics.

    `limit` caps the number of decision points (quick smoke runs). A tqdm bar
    shows live progress so a long run never looks like a silent stall. When
    `backtest.oos_start` is set, metrics are also computed separately for the
    decisions on/after that date (`oos_metrics`; None when unset/too short).
    """
    lookback = cfg["data"]["lookback"]
    pred_len = cfg["data"]["pred_len"]
    step = cfg["backtest"].get("step", 1)
    fee = cfg["backtest"].get("fee_bps", 0) / 1e4
    mom_window = cfg["backtest"].get("momentum_window", 126)
    sig_cfg = cfg["signal"]
    n_samples = 1 if fast else sig_cfg.get("sample_count", 10)

    idxs = _decision_indices(len(df), lookback, pred_len, step)
    if not idxs:
        raise ValueError("Not enough bars for even one decision; lower lookback/pred_len.")
    if limit:
        idxs = idxs[:limit]

    rows = []
    for t in tqdm(idxs, desc=f"  {ticker or 'backtest'}", unit="dec", leave=False):
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
            sig = signal_from_dist(dist, sig_cfg)
            pred_ret, action = sig.horizon_return, sig.action

        # --- realise the ACTUAL market return over the holding period (t -> t+step) ---
        # PnL is marked over `step` (we re-decide every `step` bars); the forecast
        # quality (hit-rate) is judged over the model's own horizon `pred_len`.
        close_t = float(df["close"].iloc[t])
        hold_end = min(t + step, len(df) - 1)
        actual_ret = float(df["close"].iloc[hold_end]) / close_t - 1.0
        horizon_actual = float(df["close"].iloc[t + pred_len]) / close_t - 1.0

        # Naive momentum baseline: long iff trailing `mom_window`-bar return > 0.
        mom_up = close_t > float(df["close"].iloc[max(t - mom_window, 0)])

        rows.append({
            "timestamps": df["timestamps"].iloc[t],
            "pred_return": pred_ret,
            "actual_return": actual_ret,
            "horizon_actual": horizon_actual,
            "action": action,
            "mom_action": BUY if mom_up else SELL,
        })

    res = pd.DataFrame(rows).set_index("timestamps")
    res["position"], res["strat_return"] = simulate_positions(
        res["action"], res["actual_return"], fee)
    _mom_pos, res["mom_return"] = simulate_positions(
        res["mom_action"], res["actual_return"], fee)
    res = res.drop(columns=["mom_action"])

    capital = cfg["backtest"]["initial_capital"]
    m, equity, bench, mom_curve = _compute_metrics(res, capital)

    oos_start = cfg["backtest"].get("oos_start")
    oos_m = None
    if oos_start:
        seg = res[res.index >= pd.Timestamp(oos_start)]
        if len(seg) >= 2:
            oos_m, _, _, _ = _compute_metrics(seg, capital)
        else:
            print(f"  (only {len(seg)} decision(s) after oos_start={oos_start}; "
                  f"no OOS metrics for this ticker)")

    return {"records": res, "equity": equity, "benchmark": bench,
            "momentum": mom_curve, "metrics": m, "oos_metrics": oos_m}


def plot_result(ticker: str, out: dict, out_dir: str, oos_start: str | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    equity, bench, res, m = out["equity"], out["benchmark"], out["records"], out["metrics"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity.index, equity.values, label="Kronos strategy", color="#1f77b4", lw=1.8)
    ax1.plot(bench.index, bench.values, label="Buy & hold", color="#ff7f0e", lw=1.3, alpha=0.8)
    ax1.plot(out["momentum"].index, out["momentum"].values, label="Momentum baseline",
             color="#2ca02c", lw=1.3, alpha=0.8)
    if oos_start:
        ts = pd.Timestamp(oos_start)
        if equity.index.min() <= ts <= equity.index.max():
            ax1.axvline(ts, color="gray", ls="--", lw=1.2, label="OOS start")
            ax2.axvline(ts, color="gray", ls="--", lw=1.2)
    ax1.set_ylabel("Equity")
    ax1.set_title(f"{ticker} — Kronos walk-forward backtest")
    # anchored below the stats text box so the two never overlap
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80))
    ax1.grid(True, alpha=0.3)
    txt = (f"Total: {m['total_return']:.1%}   B&H: {m['buy_hold_return']:.1%}   "
           f"Mom: {m['momentum_return']:.1%}   Excess: {m['excess_vs_bh']:+.1%}\n"
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


def _print_metrics(ticker: str, m: dict, oos: dict | None = None) -> None:
    print(f"\n=== {ticker} ===")
    print(f"  Total return   : {m['total_return']:+.2%}   (buy&hold {m['buy_hold_return']:+.2%}, "
          f"excess {m['excess_vs_bh']:+.2%})")
    print(f"  Vs momentum    : baseline {m['momentum_return']:+.2%}, "
          f"excess {m['excess_vs_mom']:+.2%}")
    print(f"  CAGR           : {m['cagr']:+.2%}")
    print(f"  Sharpe         : {m['sharpe']:.2f}")
    print(f"  Max drawdown   : {m['max_drawdown']:.2%}")
    print(f"  Directional hit: {m['hit_rate']:.2%}")
    print(f"  Win rate       : {m['win_rate']:.2%}   over {m['n_trades']} trades")
    if oos:
        print(f"  --- Out-of-sample ({oos['n_decisions']} decisions) ---")
        print(f"  OOS return     : {oos['total_return']:+.2%}   "
              f"(buy&hold {oos['buy_hold_return']:+.2%}, excess {oos['excess_vs_bh']:+.2%}; "
              f"momentum {oos['momentum_return']:+.2%}, excess {oos['excess_vs_mom']:+.2%})")
        print(f"  OOS Sharpe     : {oos['sharpe']:.2f}   MaxDD {oos['max_drawdown']:.2%}   "
              f"Hit {oos['hit_rate']:.2%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kronos walk-forward backtest")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tickers", nargs="*", help="override universe with these tickers")
    ap.add_argument("--fast", action="store_true", help="1-sample threshold signal (faster)")
    ap.add_argument("--all", action="store_true", help="include research-only tickers")
    ap.add_argument("--step", type=int, help="override backtest.step (bars between decisions)")
    ap.add_argument("--start", help="override data.history_start, e.g. 2023-01-01")
    ap.add_argument("--end", help="cap history at this date (tune here, confirm later "
                                  "on the untouched range), e.g. 2025-01-01")
    ap.add_argument("--oos-start", help="report separate out-of-sample metrics for "
                                        "decisions on/after this date")
    ap.add_argument("--limit", type=int, help="cap decisions per ticker (quick smoke run)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.step:
        cfg["backtest"]["step"] = args.step
    if args.start:
        cfg["data"]["history_start"] = args.start
    if args.end:
        cfg["data"]["history_end"] = args.end
    if args.oos_start:
        cfg["backtest"]["oos_start"] = args.oos_start
    if args.tickers:
        tickers = [(None, t, t) for t in args.tickers]
    else:
        tickers = [(r, t, n) for r, t, n, _tr in iter_universe(cfg, tradable_only=not args.all)]

    model = KronosSignalModel(cfg)
    print(f"Loaded {cfg['model']['name']} on {model.device}. "
          f"Backtesting {len(tickers)} tickers ({'fast' if args.fast else 'full'} mode).")
    calls = 1 if args.fast else cfg["signal"].get("sample_count", 10)
    print(f"Config: step={cfg['backtest'].get('step', 1)}, lookback={cfg['data']['lookback']}, "
          f"pred_len={cfg['data']['pred_len']}, history from {cfg['data'].get('history_start')}, "
          f"{calls} model call(s)/decision"
          + (f", limit={args.limit} decisions" if args.limit else "") + ".")

    summary = []
    for _region, ticker, _name in tickers:
        try:
            df = load_ticker(ticker, cfg)
            n_dec = len(_decision_indices(len(df), cfg["data"]["lookback"],
                                          cfg["data"]["pred_len"], cfg["backtest"].get("step", 1)))
            if args.limit:
                n_dec = min(n_dec, args.limit)
            print(f"\n-> {ticker}: {len(df)} bars, {n_dec} decisions "
                  f"(~{n_dec * calls} forecasts)...", flush=True)
            out = backtest_ticker(model, df, cfg, fast=args.fast, limit=args.limit, ticker=ticker)
            png = plot_result(ticker, out, cfg["backtest"]["out_dir"],
                              oos_start=cfg["backtest"].get("oos_start"))
            _print_metrics(ticker, out["metrics"], out["oos_metrics"])
            print(f"  chart -> {png}")
            row = {"ticker": ticker, **out["metrics"]}
            if out["oos_metrics"]:
                row.update({f"oos_{k}": v for k, v in out["oos_metrics"].items()})
            summary.append(row)
        except Exception as e:  # keep going across the basket
            print(f"  SKIPPED {ticker}: {e}")

    if summary:
        sdf = pd.DataFrame(summary).set_index("ticker")
        print("\n================ AGGREGATE ================")
        print(sdf[["total_return", "buy_hold_return", "momentum_return", "excess_vs_bh",
                   "excess_vs_mom", "sharpe", "max_drawdown", "hit_rate", "n_trades"]]
              .to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"\nMean excess vs buy&hold: {sdf['excess_vs_bh'].mean():+.2%}   "
              f"vs momentum: {sdf['excess_vs_mom'].mean():+.2%}   "
              f"Mean Sharpe: {sdf['sharpe'].mean():.2f}   "
              f"Mean hit-rate: {sdf['hit_rate'].mean():.2%}")
        oos_cols = [c for c in ("oos_total_return", "oos_buy_hold_return", "oos_excess_vs_bh",
                                "oos_excess_vs_mom", "oos_sharpe", "oos_hit_rate")
                    if c in sdf.columns]
        if oos_cols:
            print("\n---------- OUT-OF-SAMPLE ----------")
            print(sdf[oos_cols].to_string(float_format=lambda x: f"{x:.3f}"))
            print(f"Mean OOS excess vs buy&hold: {sdf['oos_excess_vs_bh'].mean():+.2%}   "
                  f"Mean OOS Sharpe: {sdf['oos_sharpe'].mean():.2f}")
        out_csv = os.path.join(cfg["backtest"]["out_dir"], "summary.csv")
        sdf.to_csv(out_csv)
        print(f"Summary CSV -> {out_csv}")


if __name__ == "__main__":
    main()
