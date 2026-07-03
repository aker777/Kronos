"""Information-coefficient report: is there ANY signal before you tune rules?

Reads the per-sample forecast cache written by trading/sweep.py (stage 1) and
prints, per ticker x temperature:

  * ic_h_*   — Pearson/Spearman correlation between the mean predicted return
               and the realised matched-horizon return (pred_len bars ahead).
  * ic_hold  — Spearman IC against the realised holding-period return
               (backtest.step bars), which is what PnL is marked on.
  * sign_hit — fraction of decisions where the forecast got the direction of
               the matched-horizon move right.
  * Quick PnLs of simple decision rules over the same decisions (with fees),
    next to buy-and-hold: sign (long iff mean > 0), median (+/-0.5% on the
    median sample), q25 (25th percentile > 0), tight+up (sign rule gated to
    below-median sample std — the std cutoff uses the full sample, a mild
    look-ahead, so treat it as an upper bound only).

Reading it: with n decisions, |IC| below ~2/sqrt(n) is indistinguishable from
noise — for those tickers, threshold/vote tuning is pointless by construction.
Run this after every sweep, before believing any combo table.

Usage:
    python -m trading.ic_report              # reads backtest.out_dir/sweep_cache
    python -m trading.ic_report --min-rows 50
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from trading.backtest import simulate_positions
from trading.config import load_config
from trading.signals import BUY, SELL, HOLD
from trading.sweep import _model_slug

_FNAME = re.compile(r"^(?P<ticker>.+)_T(?P<T>[0-9.]+)_p(?P<top_p>[0-9.]+)$")


def _rule_actions(kind: str, mean_r, med_r, q25, q75, std, std_med) -> list[str]:
    acts = []
    for i in range(len(mean_r)):
        a = HOLD
        if kind == "sign":
            a = BUY if mean_r[i] > 0 else SELL
        elif kind == "median":
            a = BUY if med_r[i] > 0.005 else SELL if med_r[i] < -0.005 else HOLD
        elif kind == "q25":
            a = BUY if q25[i] > 0 else SELL if q75[i] < 0 else HOLD
        elif kind == "tight+up":
            if std[i] <= std_med:
                a = BUY if mean_r[i] > 0 else SELL
        acts.append(a)
    return acts


def _pnl(actions: list[str], actual: np.ndarray, fee: float) -> float:
    _, strat = simulate_positions(actions, actual, fee)
    return float(np.prod(1.0 + np.asarray(strat)) - 1.0)


def analyze_cache_file(path: str, fee: float, min_rows: int) -> dict | None:
    name = os.path.basename(path)[: -len(".csv")]
    m = _FNAME.match(name)
    if not m:
        return None
    df = pd.read_csv(path, parse_dates=["timestamps"])
    scols = sorted((c for c in df.columns if re.fullmatch(r"s\d+", c)),
                   key=lambda c: int(c[1:]))
    if len(df) < min_rows or not scols:
        print(f"  (skipping {name}: {len(df)} rows < {min_rows} or no sample columns "
              f"— stale/partial cache, re-run the sweep)")
        return None

    s = df[scols].to_numpy()
    mean_r = s.mean(axis=1)
    med_r = np.median(s, axis=1)
    q25 = np.quantile(s, 0.25, axis=1)
    q75 = np.quantile(s, 0.75, axis=1)
    std = s.std(axis=1)
    h = df["horizon_actual"].to_numpy()      # matched pred_len-bar horizon
    hold = df["actual_return"].to_numpy()    # step-bar holding period

    pred = pd.Series(mean_r)
    row = {
        "ticker": m["ticker"], "T": m["T"], "n": len(df),
        "ic_h_pearson": pred.corr(pd.Series(h)),
        "ic_h_spearman": pred.rank().corr(pd.Series(h).rank()),
        "ic_hold": pred.rank().corr(pd.Series(hold).rank()),
        "sign_hit": float((np.sign(mean_r) == np.sign(h))[h != 0].mean()),
        "bh_return": float(np.prod(1.0 + hold) - 1.0),
    }
    std_med = float(np.median(std))
    for kind in ("sign", "median", "q25", "tight+up"):
        acts = _rule_actions(kind, mean_r, med_r, q25, q75, std, std_med)
        row[f"pnl_{kind}"] = _pnl(acts, hold, fee)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="IC report over the sweep sample cache")
    ap.add_argument("--config", default=None)
    ap.add_argument("--min-rows", type=int, default=30,
                    help="skip cache files with fewer decisions (default 30)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fee = cfg["backtest"].get("fee_bps", 0) / 1e4
    slug = _model_slug(cfg)
    cache_dir = os.path.join(cfg["backtest"]["out_dir"], "sweep_cache", slug)
    paths = sorted(glob.glob(os.path.join(cache_dir, "*.csv")))
    if not paths:
        print(f"No sweep cache for model '{slug}' in {cache_dir} — "
              f"run `python -m trading.sweep` with this model first.")
        return
    print(f"Model: {cfg['model']['name']}  (cache: {slug}, {len(paths)} files)")

    rows = [r for p in paths if (r := analyze_cache_file(p, fee, args.min_rows))]
    if not rows:
        print("Nothing usable in the cache.")
        return

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    num_cols = [c for c in out.columns if c not in ("ticker", "T", "n")]
    print("\n=== MEANS ACROSS TICKERS (per temperature) ===")
    print(out.groupby("T")[num_cols].mean().to_string(float_format=lambda x: f"{x:.3f}"))

    n_med = int(out["n"].median())
    print(f"\nNoise bar: with ~{n_med} decisions, |IC| below ~{2 / np.sqrt(n_med):.3f} "
          f"(2/sqrt(n)) is indistinguishable from zero — don't tune rules on those tickers.")
    print("PnLs are tuning-window/in-sample quick checks, not backtests; "
          "`tight+up` has mild look-ahead (upper bound).")


if __name__ == "__main__":
    main()
