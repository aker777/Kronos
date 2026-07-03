"""Turn a Kronos forecast into a trading signal.

Two rules, both configured under `signal:` in config.yaml:

  * threshold rule  — BUY if the predicted horizon return exceeds `buy_threshold`,
    SELL/exit if below `sell_threshold`, else HOLD.
  * directional-vote gate — optionally require a fraction `min_vote` of Monte-Carlo
    samples to agree on the direction before acting (more robust than one sample).
"""
from __future__ import annotations

from dataclasses import dataclass

BUY, HOLD, SELL = "BUY", "HOLD", "SELL"


@dataclass
class Signal:
    action: str          # BUY / HOLD / SELL
    horizon_return: float
    confidence: float    # up_vote for BUY, (1-up_vote) for SELL, else max side
    last_close: float
    mean_close: float


def signal_from_dist(dist: dict, signal_cfg: dict) -> Signal:
    """Build a Signal from predictor.forecast_dist() output and the signal config."""
    r = dist["horizon_return"]
    up_vote = dist["up_vote"]
    buy_th = signal_cfg["buy_threshold"]
    sell_th = signal_cfg["sell_threshold"]
    min_vote = signal_cfg.get("min_vote", 0.0)

    action = HOLD
    if r > buy_th and up_vote >= min_vote:
        action = BUY
    elif r < sell_th and (1.0 - up_vote) >= min_vote:
        action = SELL

    confidence = up_vote if action == BUY else (1.0 - up_vote) if action == SELL else max(up_vote, 1 - up_vote)
    return Signal(
        action=action,
        horizon_return=r,
        confidence=confidence,
        last_close=dist["last_close"],
        mean_close=dist["mean_close"],
    )


def signal_from_return(horizon_return: float, signal_cfg: dict) -> str:
    """Cheap threshold-only action (no vote), used for fast backtests.

    Returns BUY / HOLD / SELL.
    """
    if horizon_return > signal_cfg["buy_threshold"]:
        return BUY
    if horizon_return < signal_cfg["sell_threshold"]:
        return SELL
    return HOLD
