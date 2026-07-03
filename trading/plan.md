# Kronos Signal Toolkit — Setup, Backtest & Validation Plan

## Context

You want to turn the **Kronos** foundation model (this repo — a decoder-only
transformer that forecasts OHLCV candlesticks) into your own **signals-only**
trading tool. Universe is **global**: developed markets (Euronext Paris, Xetra,
Amsterdam, US) **plus emerging markets — Asia, Africa, and beyond**
(Hong Kong/China, India, Johannesburg, Saudi, Brazil, etc.), reached either as
individual tickers or via regional/EM ETFs. Goal: research + backtest now, with a
path to live (manual) trading later. Hardware: RTX 5080 or Colab. You place all
trades manually — the tool only emits BUY / HOLD / SELL signals.

**Broker-availability nuance:** Trade Republic / Bourse Direct are EU brokers, so
many individual African/Asian names aren't directly buyable there — in practice
you get that exposure through **EM / regional ETFs listed on Euronext/Xetra**
(MSCI EM, MSCI Asia, Africa funds, etc.). The tool still *forecasts* any
yfinance-supported global ticker (useful for research and for signalling the
underlying an ETF tracks); `config.yaml` will separate "tradable-on-my-broker"
tickers from "research-only" ones.

**Why new code:** the repo ships only *demonstration* scripts wired to Chinese
A-share data sources (`eastmoney`/`akshare`/`baostock`) with hard-coded Windows
paths, and its "backtester" (`examples/run_backtest_kronos.py`) loads
*pre-computed* prediction CSVs and aligns them loosely — it is not a true
walk-forward backtest and won't work for EU tickers. We keep the **model** and the
**metrics math**, and build a clean, English, EU-ready toolkit around them.

Decisions locked with you: **daily bars first** (data layer kept generic so
hourly can be added later); **zero-shot** pretrained Kronos first (fine-tuning
deferred until backtests justify it).

## What we reuse (do not rewrite)

- `model/` — `Kronos`, `KronosTokenizer`, `KronosPredictor`. This is the engine.
  - `KronosPredictor.predict(df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1)` returns a DataFrame of forecasted `open/high/low/close/volume/amount` indexed by `y_timestamp`. Auto-detects CUDA (`model/kronos.py:494`).
  - `max_context=512` for Kronos-small/base; keep `lookback ≤ 512`. Kronos-mini allows 2048.
- Pattern in `examples/prediction_example.py` (load models → build `x_df` + timestamps → `predict`).
- Metrics math in `examples/run_backtest_kronos.py:230` (`calculate_metrics`: total/annualised return, volatility, Sharpe, max drawdown, win rate). We port these to English and reuse the formulas.

## Deliverables — a new `trading/` package

```
trading/
  README.md            # the practical guideline (setup → data → backtest → live)
  requirements.txt     # yfinance, pyyaml, (torch installed separately, see below)
  config.yaml          # tickers, timeframe, lookback, pred_len, thresholds, model id
  data.py              # fetch/normalise OHLCV to Kronos format (timestamps,o,h,l,c,volume,amount)
  predictor.py         # thin wrapper: load model once, predict on a lookback window
  signals.py           # forecast -> signal (predicted return vs threshold; direction/vote)
  backtest.py          # TRUE walk-forward backtest over history + metrics + plot
  live_signal.py       # fetch latest bars -> today's BUY/HOLD/SELL per ticker (manual exec)
```

## Implementation steps

### 1. Environment (venv + PyTorch for RTX 5080)
- `python -m venv .venv` and activate.
- **RTX 5080 is Blackwell (sm_120)** — stock `torch` wheels fail. Install from the CUDA 12.8 index: `pip install torch --index-url https://download.pytorch.org/whl/cu128`. On Colab, the preinstalled torch already supports it; just `pip install -r requirements.txt -r trading/requirements.txt`.
- Then `pip install -r requirements.txt` (repo core) + `trading/requirements.txt` (`yfinance`, `pyyaml`).
- Sanity check: `python -c "import torch; print(torch.cuda.is_available())"`.

### 2. `data.py` — EU-ready data layer (generic over timeframe)
- Use **`yfinance`** (free, genuinely global). Suffix coverage the config will use:
  developed — `.PA` Paris, `.DE` Xetra, `.AS` Amsterdam, `.MI` Milan, `.L` London, US (no suffix);
  **emerging/Asia** — `.HK` Hong Kong, `.SS`/`.SZ` Shanghai/Shenzhen, `.NS`/`.BO` India, `.KS` Korea, `.TW` Taiwan, `.JK` Indonesia;
  **Africa/MEA** — `.JO` Johannesburg, `.SR` Saudi, `.CA` Egypt;
  **LatAm** — `.SA` Brazil, `.MX` Mexico; plus EM/regional **ETFs** (e.g. `EIMI.L`, `EUNM.DE`).
- Function `fetch_ohlcv(ticker, interval, start, end)` with `interval` in `{"1d","1h",...}` so daily works now and hourly later with no API change.
- Map yfinance columns → Kronos schema exactly: `timestamps, open, high, low, close, volume, amount`. Set `amount = 0` (Kronos tolerates it). Drop NaNs, sort ascending, tz-normalise.
- **EM data-quality guards** (thin/illiquid markets misbehave): drop tickers with too few bars or long gaps, warn on zero-volume stretches and stale (repeated) closes, and handle local-holiday gaps. `config.yaml` groups tickers by region and flags each as `tradable` vs `research_only`.
- Cache to `trading/data_cache/*.csv` (already git-ignored: `*.csv` is in `.gitignore`).

### 3. `predictor.py` — load-once wrapper
- Load tokenizer + model from HF **once** (`NeoQuasar/Kronos-Tokenizer-base` + `NeoQuasar/Kronos-small` default; `Kronos-base` optional via config). Instantiate `KronosPredictor(max_context=512)`.
- `forecast(df_window, pred_len, sample_count, T, top_p)` → returns predicted OHLCV. `sample_count>1` averages Monte-Carlo samples (reduces noise; the model is stochastic).

### 4. `signals.py` — forecast → signal
- Core rule (port/clean from `run_backtest_kronos.py:132`): compute predicted return over the horizon `= pred_close[-1]/last_actual_close - 1`; BUY if `> +threshold`, SELL/exit if `< -threshold`, else HOLD.
- Add a **directional-vote** variant using `sample_count` samples (fraction of samples predicting up) for a confidence score — more robust than a single sample.
- Keep thresholds in `config.yaml` so they're tunable in validation.

### 5. `backtest.py` — TRUE walk-forward (the core of "testing & validating")
- For each step `t` over a held-out history: take the trailing `lookback` window ending at `t`, call `predictor.forecast(...)`, derive a signal, then advance and mark-to-market against **actual** future closes. This is the honest test the repo lacks.
- Long/flat only (matches signals-only, no shorting on TR/Bourse Direct retail).
- Report, per ticker and aggregate: total & annualised return, Sharpe, max drawdown, win rate, **directional hit-rate**, number of trades, and **benchmark vs buy-and-hold**. Port formulas from `calculate_metrics`.
- Save an equity-curve + drawdown PNG (reuse the plotting shape from `run_backtest_kronos.py:293`, English labels).
- **Validation protocol baked in:** out-of-sample split (don't tune on test window); sweep `threshold`, `T`, `top_p`, `sample_count`; run across several tickers (a few ETFs + stocks) so results aren't one lucky symbol. Compare against buy-and-hold and a naive momentum baseline — Kronos must beat both to be worth trading.

### 6. `live_signal.py` — daily manual signal
- Fetch the latest `lookback` bars per configured ticker, forecast next `pred_len`, print a table: ticker, last close, predicted return, signal, confidence. Optionally append to `trading/signals_log.csv`. You then place trades yourself.

### 7. `trading/README.md` — the guideline
- End-to-end runbook: install → configure tickers → run backtest → read metrics → generate daily signals. Includes the RTX 5080 / Colab notes and a plain-English "how to judge if this is worth trading" section (beat buy-and-hold after costs, positive Sharpe out-of-sample, stable across tickers). Flag realism caveats: model trained largely on non-EU data (zero-shot), transaction costs/spread, and that forecasts are stochastic.

## Fine-tuning (deferred — documented, not built now)
Once backtests look promising, adapt Kronos to EU data with the existing
`finetune_csv/` pipeline (`train_sequential.py --config ...`, needs a YAML like
`configs/config_ali09988_candle-5min.yaml` pointed at an EU CSV). RTX 5080 handles
small/base; use Colab for larger runs. README will point here.

## Verification (how we'll confirm it works)
1. **Smoke test the engine:** run a one-shot forecast on one ticker (e.g. `CW8.PA` or `AAPL`) via `predictor.py` and confirm a valid OHLCV DataFrame comes back on GPU — mirrors `examples/prediction_example.py` but with yfinance data.
2. **Data layer:** assert `fetch_ohlcv` returns the exact Kronos columns, ascending timestamps, no NaNs, across regions — a `.PA` ETF, a US stock, and at least one **emerging-market** ticker (e.g. `.HK`, `.NS`, or `.JO`); confirm the EM data-quality guards fire on a thin ticker.
3. **Backtest end-to-end:** run `backtest.py` on a multi-region basket (developed + EM, incl. an Asia and an Africa name) over a multi-year daily window; confirm it produces per-ticker + aggregate metrics, equity-curve PNG, and a buy-and-hold comparison. Eyeball that the walk-forward uses only past data at each step (no look-ahead).
4. **Live signal:** run `live_signal.py`; confirm it prints a current signal table without error.
5. Existing repo tests still pass: `python -m pytest tests/` (regression test `tests/test_kronos_regression.py` — unaffected, but a guard against breaking the model package).

## Commit (next action — approve to execute)
The toolkit is built and validated. On approval I will:
1. Create a branch off `master`: `feature/kronos-signal-toolkit`.
2. Stage **only** the new `trading/` package and the `.gitignore` change
   (the `.venv/` and generated outputs are git-ignored; the pre-existing
   unrelated `*.pyc`/`*.csv` lines already in `.gitignore` come along as part of
   that one file).
3. Commit with message: `Add Kronos signal toolkit (data, forecasting, backtest, live signals)`.
4. Push with `git push -u origin feature/kronos-signal-toolkit` (retry with backoff on network errors).
No pull request will be opened (not requested).

## Follow-up: backtest progress + quick-run UX (approve to execute)
Problem observed on the user's Windows machine: `python -m trading.backtest --fast`
prints the "Backtesting 8 tickers" header then goes silent for a long time —
each ticker runs ~2000 sequential GPU forecasts (step=1, history from 2015) and
results only print after a ticker fully completes. Looks hung; isn't.

Changes (all small, in `trading/backtest.py`):
1. **Per-ticker progress**: before the heavy loop, print
   `-> {ticker}: {n} bars, {N} decisions...`; wrap the decision loop in a `tqdm`
   bar (tqdm is already a repo dependency) for live feedback.
2. **CLI quick-run overrides** (no config editing): add `--step N`,
   `--start YYYY-MM-DD`, and `--limit N` (cap decisions), so a fast smoke run is
   one command, e.g. `--fast --tickers AAPL --step 20 --start 2023-01-01`.
3. **Startup hint**: print estimated decisions/ticker after load so the user
   knows up front whether a run will be long.
Verification: `--fast --tickers AAPL --step 20 --start 2023-01-01` shows progress
bar + metrics + PNG quickly; a default single-ticker run shows the bar advancing
(no silent stall).

- yfinance intraday history for EU is limited (~730 days of 1h); daily is unlimited — daily is the primary path as agreed.
- Zero-shot Kronos on EU daily bars is unproven; the backtest is precisely what tells us if it's tradable. No live money until out-of-sample results beat buy-and-hold after costs.