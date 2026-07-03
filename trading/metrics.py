"""Performance metrics for the walk-forward backtest.

Formulas are ported to English from examples/run_backtest_kronos.py:230
(`calculate_metrics`) and generalised to work off a per-decision returns series
indexed by timestamp, so they annualise correctly for daily or hourly data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _years(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    days = (index[-1] - index[0]).days
    return max(days / 365.25, 1e-9)


def summarize(returns: pd.Series, equity: pd.Series, risk_free: float = 0.0) -> dict:
    """Compute headline metrics from a per-decision returns series and equity curve.

    Args:
        returns: fractional return realised at each decision point (indexed by time).
        equity:  cumulative capital at each decision point (same index).
        risk_free: annual risk-free rate for Sharpe.
    """
    returns = returns.dropna()
    if len(returns) == 0 or equity.empty:
        return {k: 0.0 for k in
                ("total_return", "cagr", "volatility", "sharpe", "max_drawdown", "n_decisions")}

    years = _years(returns.index)
    per_year = len(returns) / years                      # decisions per year

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0
    volatility = float(returns.std(ddof=0) * np.sqrt(per_year))
    ann_return = returns.mean() * per_year
    sharpe = float((ann_return - risk_free) / volatility) if volatility > 0 else 0.0

    curve = (1.0 + returns).cumprod()
    peak = curve.cummax()
    max_dd = float(((curve - peak) / peak).min())

    return {
        "total_return": total_return,
        "cagr": float(cagr),
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_decisions": int(len(returns)),
    }


def trade_stats(position: pd.Series, period_return: pd.Series) -> dict:
    """Win rate and trade count from a long/flat position path.

    A "trade" is a stretch of consecutive long bars; its return is the compounded
    market return over that stretch.
    """
    pos = position.fillna(0).values
    ret = period_return.fillna(0).values
    trades: list[float] = []
    in_trade = False
    cur = 1.0
    for p, r in zip(pos, ret):
        if p > 0:
            cur *= (1.0 + r)
            in_trade = True
        elif in_trade:
            trades.append(cur - 1.0)
            cur, in_trade = 1.0, False
    if in_trade:
        trades.append(cur - 1.0)

    n = len(trades)
    win_rate = float(np.mean([t > 0 for t in trades])) if n else 0.0
    avg_trade = float(np.mean(trades)) if n else 0.0
    return {"n_trades": n, "win_rate": win_rate, "avg_trade_return": avg_trade}


def directional_hit_rate(pred_return: pd.Series, actual_return: pd.Series) -> float:
    """Fraction of decisions where the forecast got the direction right."""
    mask = pred_return.notna() & actual_return.notna() & (actual_return != 0)
    if mask.sum() == 0:
        return 0.0
    return float((np.sign(pred_return[mask]) == np.sign(actual_return[mask])).mean())
