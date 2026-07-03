"""Parameter sweep for the Kronos signal toolkit.

Searches threshold / min_vote / sample_count / temperature / top_p combinations
without re-running the model for every combo:

  Stage 1 (model, slow): per ticker and per (T, top_p) pair, walk the same
    decision grid as the backtest ONCE, calling forecast_dist with the LARGEST
    sample_count in the grid, and cache every decision's per-sample horizon
    returns (plus realised forward returns) under
    trading/backtest_results/sweep_cache/.
  Stage 2 (offline, instant): every (threshold, min_vote, sample_count) combo is
    rebuilt from the stored samples — signals via signals.signal_from_dist, PnL
    via backtest.simulate_positions — so signal & PnL rules cannot drift from
    the real backtest.

Rank on the OUT-OF-SAMPLE columns (--oos-start): picking the best fully
in-sample combo is overfitting by construction. The grid lives under `sweep:` in
config.yaml.

Usage:
    python -m trading.sweep --tickers AAPL --step 20 --start 2024-01-01 --limit 30
    python -m trading.sweep --tickers AAPL CW8.PA --oos-start 2025-06-01
"""
from __future__ import annotations

import argparse
import itertools
import os
import re

import pandas as pd
from tqdm import tqdm

from trading import metrics
from trading.backtest import _decision_indices, simulate_positions
from trading.config import load_config, iter_universe
from trading.data import load_ticker
from trading.predictor import KronosSignalModel
from trading.signals import signal_from_dist

_DEFAULT_GRID = {
    "thresholds": [0.005, 0.01, 0.02],
    "min_votes": [0.5, 0.6, 0.7],
    "T": [0.8, 1.0],
    "top_p": [0.9],
    "sample_counts": [5, 10],
}


def _model_slug(cfg: dict) -> str:
    """Short directory-safe tag for the configured model, so caches from
    different checkpoints (zero-shot vs fine-tuned) can never mix.

    Local checkpoints all end in generic dir names (basemodel/best_model), so
    for those the experiment directory is folded in: e.g.
    finetune_csv/finetuned/CW8_PA_1d/basemodel/best_model
    -> CW8_PA_1d_basemodel_best_model. Hub ids keep their basename
    (NeoQuasar/Kronos-small -> Kronos-small).
    """
    name = str(cfg["model"]["name"]).rstrip("/\\")
    parts = re.split(r"[/\\]+", name)
    base = parts[-1]
    if base.lower() in ("best_model", "checkpoint", "model") and len(parts) >= 3:
        base = "_".join(parts[-3:])
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def _cache_path(out_dir: str, model_slug: str, ticker: str, T: float, top_p: float) -> str:
    safe = ticker.replace("/", "_").replace(".", "_")
    return os.path.join(out_dir, "sweep_cache", model_slug, f"{safe}_T{T}_p{top_p}.csv")


def collect_samples(model: KronosSignalModel, df: pd.DataFrame, cfg: dict,
                    T: float, top_p: float, n_samples: int,
                    limit: int | None, ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Stage 1: one walk over the decision grid storing per-sample horizon returns.

    Reuses the CSV cache when it covers the same decision grid with at least
    `n_samples` samples; `refresh=True` forces a re-run.
    """
    lookback = cfg["data"]["lookback"]
    pred_len = cfg["data"]["pred_len"]
    step = cfg["backtest"].get("step", 1)

    idxs = _decision_indices(len(df), lookback, pred_len, step)
    if not idxs:
        raise ValueError("Not enough bars for even one decision; lower lookback/pred_len.")
    if limit:
        idxs = idxs[:limit]

    path = _cache_path(cfg["backtest"]["out_dir"], _model_slug(cfg), ticker, T, top_p)
    want_first = pd.Timestamp(df["timestamps"].iloc[idxs[0]])
    want_last = pd.Timestamp(df["timestamps"].iloc[idxs[-1]])
    if os.path.exists(path) and not refresh:
        cached = pd.read_csv(path, parse_dates=["timestamps"])
        n_cached = sum(c.startswith("s") and c[1:].isdigit() for c in cached.columns)
        if (len(cached) == len(idxs) and n_cached >= n_samples
                and cached["timestamps"].iloc[0] == want_first
                and cached["timestamps"].iloc[-1] == want_last):
            return cached

    rows = []
    for t in tqdm(idxs, desc=f"  {ticker} T={T} top_p={top_p}", unit="dec", leave=False):
        dist = model.forecast_dist(df, end_idx=t, n_samples=n_samples, T=T, top_p=top_p)
        close_t = float(df["close"].iloc[t])
        hold_end = min(t + step, len(df) - 1)
        row = {
            "timestamps": df["timestamps"].iloc[t],
            "last_close": close_t,
            "actual_return": float(df["close"].iloc[hold_end]) / close_t - 1.0,
            "horizon_actual": float(df["close"].iloc[t + pred_len]) / close_t - 1.0,
        }
        row.update({f"s{i}": float(r) for i, r in enumerate(dist["sample_returns"])})
        rows.append(row)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return out


def _segment_metrics(res: pd.DataFrame, capital: float) -> dict:
    equity = capital * (1.0 + res["strat_return"]).cumprod()
    m = metrics.summarize(res["strat_return"], equity)
    m["total_return"] = float(equity.iloc[-1] / capital - 1.0)
    bench_total = float((1.0 + res["actual_return"]).prod() - 1.0)
    m["excess_vs_bh"] = m["total_return"] - bench_total
    m["hit_rate"] = metrics.directional_hit_rate(res["pred_return"], res["horizon_actual"])
    m["n_trades"] = metrics.trade_stats(res["position"], res["actual_return"])["n_trades"]
    keep = ("total_return", "sharpe", "max_drawdown", "excess_vs_bh",
            "hit_rate", "n_trades", "n_decisions")
    return {k: m[k] for k in keep}


def evaluate_combo(samples: pd.DataFrame, thr: float, vote: float, k: int,
                   fee: float, capital: float, oos_start: str | None) -> dict:
    """Stage 2: rebuild signals from the first `k` stored samples and simulate."""
    sub = samples[[f"s{i}" for i in range(k)]].to_numpy()
    mean_ret = sub.mean(axis=1)          # == mean_close/last_close - 1 (linear)
    up_vote = (sub > 0).mean(axis=1)
    sig_cfg = {"buy_threshold": thr, "sell_threshold": -thr, "min_vote": vote}

    actions = []
    for r, v, lc in zip(mean_ret, up_vote, samples["last_close"]):
        dist = {"horizon_return": float(r), "up_vote": float(v),
                "last_close": float(lc), "mean_close": float(lc) * (1.0 + float(r))}
        actions.append(signal_from_dist(dist, sig_cfg).action)

    positions, strat = simulate_positions(actions, samples["actual_return"].to_numpy(), fee)
    res = pd.DataFrame(
        {
            "pred_return": mean_ret,
            "actual_return": samples["actual_return"].to_numpy(),
            "horizon_actual": samples["horizon_actual"].to_numpy(),
            "position": positions,
            "strat_return": strat,
        },
        index=pd.DatetimeIndex(pd.to_datetime(samples["timestamps"])),
    )

    m = _segment_metrics(res, capital)
    if oos_start:
        seg = res[res.index >= pd.Timestamp(oos_start)]
        if len(seg) >= 2:
            m.update({f"oos_{key}": v for key, v in _segment_metrics(seg, capital).items()})
    return m


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kronos parameter sweep (T/top_p via model passes; "
                    "thresholds/votes/sample_counts offline)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tickers", nargs="*", help="override universe with these tickers")
    ap.add_argument("--step", type=int, help="override backtest.step")
    ap.add_argument("--start", help="override data.history_start")
    ap.add_argument("--end", help="cap history at this date")
    ap.add_argument("--oos-start", help="rank on out-of-sample metrics from this date")
    ap.add_argument("--limit", type=int, help="cap decisions per ticker (smoke run)")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the sweep cache and re-run the model")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.step:
        cfg["backtest"]["step"] = args.step
    if args.start:
        cfg["data"]["history_start"] = args.start
    if args.end:
        cfg["data"]["history_end"] = args.end

    grid = {**_DEFAULT_GRID, **(cfg.get("sweep") or {})}
    fee = cfg["backtest"].get("fee_bps", 0) / 1e4
    capital = cfg["backtest"]["initial_capital"]
    oos_start = args.oos_start or cfg["backtest"].get("oos_start")
    if not oos_start:
        print("WARNING: no --oos-start given — every number below is IN-SAMPLE; "
              "picking the best combo from this table alone is overfitting.")

    if args.tickers:
        tickers = args.tickers
    else:
        tickers = [t for _r, t, _n, _tr in iter_universe(cfg, tradable_only=True)]

    max_k = max(grid["sample_counts"])
    n_offline = len(grid["thresholds"]) * len(grid["min_votes"]) * len(grid["sample_counts"])
    model = KronosSignalModel(cfg)
    print(f"Loaded {cfg['model']['name']} on {model.device}. "
          f"Stage 1: {len(tickers) * len(grid['T']) * len(grid['top_p'])} decision-grid "
          f"pass(es) at {max_k} samples/decision; Stage 2: {n_offline} offline combos each.")

    rows = []
    for ticker in tickers:
        try:
            df = load_ticker(ticker, cfg)
        except Exception as e:  # keep going across the basket
            print(f"  SKIPPED {ticker}: {e}")
            continue
        for T, top_p in itertools.product(grid["T"], grid["top_p"]):
            try:
                samples = collect_samples(model, df, cfg, T, top_p, max_k,
                                          args.limit, ticker, refresh=args.refresh)
            except Exception as e:
                print(f"  SKIPPED {ticker} (T={T}, top_p={top_p}): {e}")
                continue
            for thr, vote, k in itertools.product(
                    grid["thresholds"], grid["min_votes"], grid["sample_counts"]):
                m = evaluate_combo(samples, thr, vote, k, fee, capital, oos_start)
                rows.append({"ticker": ticker, "T": T, "top_p": top_p, "threshold": thr,
                             "min_vote": vote, "sample_count": k, **m})

    if not rows:
        print("No results — nothing to rank.")
        return

    full = pd.DataFrame(rows)
    out_dir = cfg["backtest"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    full_csv = os.path.join(out_dir, "sweep_results.csv")
    full.to_csv(full_csv, index=False)

    combo_cols = ["T", "top_p", "threshold", "min_vote", "sample_count"]
    agg = full.groupby(combo_cols).mean(numeric_only=True)
    agg["n_tickers"] = full.groupby(combo_cols)["ticker"].nunique()
    rank_col = "oos_sharpe" if "oos_sharpe" in agg.columns else "sharpe"
    agg = agg.sort_values(rank_col, ascending=False)

    show = [c for c in ("sharpe", "excess_vs_bh", "hit_rate", "total_return",
                        "oos_sharpe", "oos_excess_vs_bh", "oos_hit_rate", "n_tickers")
            if c in agg.columns]
    print(f"\n===== TOP COMBOS (mean across {full['ticker'].nunique()} ticker(s), "
          f"ranked by {rank_col}) =====")
    print(agg[show].head(15).to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nFull per-ticker results -> {full_csv}")
    print("Reminder: confirm any chosen combo with a full backtest on data the "
          "sweep never saw (--oos-start / --end split).")


if __name__ == "__main__":
    main()
