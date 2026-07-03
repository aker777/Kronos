"""Load and expose the toolkit configuration (trading/config.yaml)."""
from __future__ import annotations

import os
import yaml

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path: str | None = None) -> dict:
    """Read config.yaml into a plain dict."""
    with open(path or _DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def iter_universe(cfg: dict, tradable_only: bool = False):
    """Yield (region, ticker, name, tradable) across every group in `universe`.

    Args:
        tradable_only: if True, skip research-only tickers.
    """
    for region, entries in (cfg.get("universe") or {}).items():
        for entry in entries:
            tradable = bool(entry.get("tradable", True))
            if tradable_only and not tradable:
                continue
            yield region, entry["ticker"], entry.get("name", entry["ticker"]), tradable


def all_tickers(cfg: dict, tradable_only: bool = False) -> list[str]:
    """Flat list of tickers across the universe."""
    return [t for _region, t, _name, _tr in iter_universe(cfg, tradable_only)]
