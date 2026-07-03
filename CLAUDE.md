# CLAUDE.md — Kronos + Signal Toolkit

Context for Claude sessions working on this repo. Read this first.

## What this repo is
**Kronos** — an open-source decoder-only transformer foundation model that
forecasts financial OHLCV candlesticks (K-lines). The model, tokenizer and
predictor live in `model/` and are pulled as pretrained weights from Hugging Face
(`NeoQuasar/Kronos-*`). Upstream: https://github.com/shiyu-coder/Kronos

## What we added: `trading/` (the user's actual goal)
A **signals-only** trading research toolkit built on top of Kronos. The user
(France; brokers Trade Republic & Bourse Direct; RTX 5080 or Colab) wants to
research/backtest a global universe (developed + emerging: Asia, Africa, LatAm),
emit BUY/HOLD/SELL signals, and place trades manually. Zero-shot first;
fine-tuning deferred.

```
trading/
  config.yaml      # model, timeframe(1d), lookback/pred_len, thresholds, region-grouped universe
  config.py        # config loader + iter_universe/all_tickers helpers
  data.py          # yfinance -> Kronos schema (timestamps,o,h,l,c,volume,amount) + quality guards
  predictor.py     # KronosSignalModel: load model once; forecast() + forecast_dist() (vote)
  signals.py       # forecast -> BUY/HOLD/SELL (threshold + directional-vote gate)
  metrics.py       # Sharpe, max drawdown, hit-rate, win-rate (ported from run_backtest_kronos.py)
  backtest.py      # TRUE walk-forward backtest (no look-ahead) + baselines + OOS split + charts
  sweep.py         # parameter sweep: T/top_p via model passes; thresholds/votes offline from cache
  ic_report.py     # IC (forecast-vs-realised correlation) per ticker from the sweep cache
  live_signal.py   # today's signals table for manual execution
  README.md        # full setup + validation runbook
```

### Key design facts (don't relearn the hard way)
- **Core API**: `KronosPredictor.predict(df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p, sample_count)`
  returns forecast OHLCV. `predict()` **averages** `sample_count` Monte-Carlo
  samples internally (`model/kronos.py:467`) — so for a directional *vote* we run
  repeated single-sample forecasts in `predictor.forecast_dist()`.
- **Data schema is exact**: `timestamps, open, high, low, close, volume, amount`.
  yfinance has no turnover → `amount = 0` (Kronos tolerates it).
- **Context limit**: `max_context=512` for Kronos-small/base; keep `lookback <= 512`.
- **Walk-forward is honest**: at each decision `t`, only bars `<= t` are fed to
  the model; PnL is marked over `step`; hit-rate is judged over the forecast's own
  `pred_len` horizon (`horizon_actual` column in `backtest.py`).
- **Backtest cost**: `fee_bps` applied on position changes; long/flat only (no shorting).
- **Baselines**: every backtest reports buy-and-hold AND a naive momentum baseline
  (long iff trailing `momentum_window`-bar return > 0, same fees). Kronos must beat both.
- **PnL logic is shared**: `backtest.simulate_positions` is used by both the backtest
  and `sweep.py` stage 2 — change position/fee rules there only.
- **Sweep efficiency**: only `T`×`top_p` combos need model passes; thresholds/min_vote/
  sample_count are re-evaluated offline from per-sample returns cached in
  `trading/backtest_results/sweep_cache/`.

## Running it (user is on Windows 10 + RTX 5080)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell/sm_120; skip on Colab
pip install -r requirements.txt -r trading\requirements.txt

# fast, visible smoke test (progress bar + one ticker + short history)
python -m trading.backtest --fast --tickers AAPL --step 20 --start 2023-01-01
# full read (10-sample vote) once smoke test passes
python -m trading.backtest
python -m trading.live_signal
```
CLI flags on `backtest.py`: `--fast` (1-sample threshold), `--all` (include
research-only tickers), `--step N`, `--start/--end YYYY-MM-DD`, `--oos-start
YYYY-MM-DD` (separate out-of-sample metrics), `--limit N`, `--tickers ...`.
`sweep.py` mirrors them plus `--refresh` (ignore the sample cache).

Validation workflow: tune with `--end CUTOFF` (backtest + sweep), then confirm
once with `--oos-start CUTOFF` on the untouched range. **IC first**: run
`python -m trading.ic_report` after a sweep — tickers with |IC| under the
2/sqrt(n) noise bar have no extractable signal; don't tune rules on them.
Campaign result 2026-07: developed markets IC ≈ 0 (dead end); EM names are the
only ones showing IC above noise. Data cache self-widens (sidecar .meta files)
and always fetches to the present, so `--end` runs can't truncate it.

### Gotchas
- Run as a module (`python -m trading.backtest`) from repo root — not `python trading\backtest.py`.
- First run downloads ~100 MB of weights from Hugging Face (needs internet, once).
- If `torch.cuda.is_available()` is False on the 5080: driver/wheel mismatch — use the cu128 index above and update the NVIDIA driver (R570+).
- yfinance intraday (`1h`) EU history is limited (~730 days); daily is unlimited and the primary path.

## Repo / git state
- Branch: `feature/kronos-signal-toolkit`. Commits: `4b8e79b` (toolkit),
  `14c31b4` (backtest progress + quick-run flags).
- User's fork: `https://github.com/aker777/Kronos`. Local sessions on the user's
  machine can `git push` directly; only web/container sessions lack push access
  (there, deliver changes as `git format-patch`).
- Commit trailers required in this environment (chat-only; never in code):
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` + `Claude-Session:` line.

## How to judge if the strategy is worth trading
Beats buy-and-hold **after** `fee_bps`, positive Sharpe out-of-sample,
directional hit-rate >50%, and **stable across many tickers/regions** (not one
lucky symbol). No live money until that holds on a full multi-year run.

## Fine-tuning experiment (set up, not yet launched)
`python -m trading.export_finetune` prepares per-ticker predictor-only
fine-tunes (CSV ≤ 2025-07-01 in `finetune_csv/data/`, configs in
`finetune_csv/configs/config_{TICKER}_1d.yaml`, pretrained weights snapshotted
to `finetune_csv/pretrained/`). Launch from `finetune_csv\`:
`..\.venv\Scripts\python.exe train_sequential.py --config configs/config_CW8_PA_1d.yaml --skip-tokenizer`
Then evaluate per README §5: point `model.name` at the checkpoint, sweep with
`--end 2025-07-01`, `ic_report` (gate: IC above noise), backtest `--oos-start`.
Sweep caches are tagged per model slug (`sweep_cache/<model>/`).

(Done earlier: sweep, OOS split, momentum baseline, IC report. Zero-shot failed
the gate on developed AND EM names — see git history for both campaigns.)

## Constraints when validating changes in the cloud container
The web/container environment **blocks egress** to `huggingface.co`,
`download.pytorch.org`, and `finance.yahoo.com` (all 403). So the real
weight-load / forecast / live-fetch can only be verified on the user's machine.
In-container, validate logic with: mocked `yfinance` for the data transform, and a
stub/oracle model injected into `backtest_ticker` for the walk-forward mechanics.
