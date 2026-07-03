# Kronos Signal Toolkit

A **signals-only** trading research tool built on the [Kronos](../README.md)
candlestick foundation model. It fetches global OHLCV data (Europe, US, and
emerging markets across Asia, Africa and LatAm), forecasts the next few bars with
a pretrained Kronos model **zero-shot**, turns each forecast into a
**BUY / HOLD / SELL** signal, and — most importantly — **backtests those signals
walk-forward** so you can judge whether they are actually worth trading.

You place every trade yourself (e.g. on Trade Republic / Bourse Direct). The tool
never touches a broker.

```
trading/
  config.yaml      # model, timeframe, horizon, thresholds, ticker universe
  config.py        # config loader + universe helpers
  data.py          # yfinance -> Kronos schema, with data-quality guards
  predictor.py     # load Kronos once; forecast a lookback window
  signals.py       # forecast -> BUY/HOLD/SELL (threshold + directional vote)
  metrics.py       # Sharpe, max drawdown, hit-rate, win-rate, buy&hold
  backtest.py      # TRUE walk-forward backtest + baselines + charts
  sweep.py         # parameter sweep (thresholds/T/top_p/sample_count)
  ic_report.py     # information-coefficient report over the sweep cache
  live_signal.py   # today's signals for the universe
```

---

## 1. Install

Create a virtual environment and install PyTorch **first** (it must match your
hardware), then the rest.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# --- PyTorch ---
# RTX 50-series (5080 = Blackwell / sm_120): the default wheels DO NOT work.
pip install torch --index-url https://download.pytorch.org/whl/cu128
# CPU-only (small tests, no GPU):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
# Google Colab: torch is preinstalled and already GPU-capable — skip this line.

# --- the rest ---
pip install -r requirements.txt          # repo core (from the project root)
pip install -r trading/requirements.txt  # yfinance, pyyaml
```

Sanity check the GPU:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

On the **RTX 5080** the small/base models fit comfortably. On **Colab**, pick a
GPU runtime; a T4 is enough for `Kronos-small`/`Kronos-base`.

---

## 2. Configure

Edit `trading/config.yaml`:

- **`model.name`** — `Kronos-small` (fast default), `Kronos-base` (stronger), or
  `Kronos-mini` (long 2048 context). Keep `max_context` matched (512 for
  small/base, 2048 for mini).
- **`data.interval`** — `"1d"` daily (primary) or `"1h"` hourly. Daily has
  unlimited history; yfinance only serves ~730 days of EU hourly data.
- **`data.lookback` / `data.pred_len`** — bars fed to the model / bars forecast
  ahead. Keep `lookback <= max_context`.
- **`signal.*`** — buy/sell thresholds, Monte-Carlo `sample_count`, and the
  `min_vote` confidence gate.
- **`universe`** — grouped by region. Each ticker is flagged `tradable: true`
  (directly buyable on your broker) or `tradable: false` (research-only — you get
  that exposure via an EM/regional ETF instead). yfinance suffixes:
  `.PA` Paris · `.DE` Xetra · `.AS` Amsterdam · `.L` London · US = no suffix ·
  `.HK` Hong Kong · `.NS` India · `.KS` Korea · `.JO` Johannesburg · `.SR` Saudi ·
  `.SA` Brazil.

---

## 3. Backtest (do this before trusting any signal)

```bash
# whole tradable universe, full Monte-Carlo signal
python -m trading.backtest

# quick pass (1 sample, threshold-only) — much faster while you tune
python -m trading.backtest --fast

# specific tickers, including research-only names
python -m trading.backtest --tickers AAPL CW8.PA 0700.HK NPN.JO --all

# quick smoke run
python -m trading.backtest --fast --tickers AAPL --step 20 --start 2023-01-01
```

For each ticker you get an equity-curve + drawdown PNG in
`trading/backtest_results/` (strategy vs **buy & hold** vs a naive
**momentum baseline** — long iff the trailing `momentum_window`-bar return is
positive, same fees), plus a metrics table and an aggregate summary CSV.
The backtest is **walk-forward**: at each step the model sees only past bars, and
the signal is marked against the *actual* future move — no look-ahead.

### Out-of-sample validation (the honest protocol)

Never judge a parameter choice on the data you tuned it on:

```bash
# 1. tune on history up to a cutoff only
python -m trading.backtest --end 2025-01-01
python -m trading.sweep    --end 2025-01-01

# 2. confirm ONCE on the untouched range (separate OOS metrics are reported)
python -m trading.backtest --oos-start 2025-01-01
```

`--oos-start` prints a per-ticker `OUT-OF-SAMPLE` block, adds `oos_*` columns to
`summary.csv`, and draws the split line on the charts.

### Parameter sweep

```bash
python -m trading.sweep --tickers AAPL CW8.PA --oos-start 2025-01-01
python -m trading.sweep --tickers AAPL --step 20 --start 2024-01-01 --limit 30  # smoke
```

The grid lives under `sweep:` in `config.yaml`. Only `T` × `top_p` combinations
need model passes; every `threshold` / `min_vote` / `sample_count` combination is
re-evaluated instantly from cached per-sample returns
(`trading/backtest_results/sweep_cache/`, reused across runs — `--refresh` to
force). Output is a ranked combo table plus `sweep_results.csv`. Rank on the
`oos_*` columns; the best purely in-sample combo is overfit by construction.

### IC first — before believing any combo table

```bash
python -m trading.ic_report        # after a sweep
```

Prints, per ticker, the **information coefficient** (correlation between the
forecast and what actually happened) with its 2/√n noise bar. If a ticker's
|IC| is below the noise bar, Kronos has **no signal there** and no threshold or
vote setting can create one — remove the ticker or change the model, don't keep
tuning. (Empirically, developed-market tickers sit at IC ≈ 0; EM names are
where any signal has appeared.)

### How to judge if it's worth trading
Look for **all** of these, not just a big total return:
- **Beats buy & hold** (`excess_vs_bh > 0`) *after* the `fee_bps` costs.
- **Beats the momentum baseline** (`excess_vs_mom > 0`) — otherwise a dumb
  moving-average rule earns the same without a GPU.
- **Positive Sharpe** and a tolerable **max drawdown**.
- **Directional hit-rate > 50%** — the forecast is genuinely informative.
- **Stable across many tickers and regions**, not one lucky symbol.
- Holds up **out-of-sample** (`--oos-start`), not just on the tuning window.

If it doesn't clear that bar zero-shot, that's a real result — consider
fine-tuning (below) or a different horizon before risking money.

---

## 4. Live (manual) signals

```bash
python -m trading.live_signal            # tradable universe
python -m trading.live_signal --all      # include research-only tickers
python -m trading.live_signal --log      # also append to signals_log.csv
```

Prints a table of ticker, last close, predicted horizon return, signal and
confidence. Trades are yours to place.

---

## 5. Fine-tuning experiment (set up — launch when ready)

Two zero-shot campaigns failed the OOS gate, so fine-tuning is the remaining
lever that could change the answer (it changes the model, not the rules).
Everything is prepared by:

```bash
python -m trading.export_finetune                    # CW8.PA, EUNM.DE, AAPL
python -m trading.export_finetune --tickers MC.PA    # any other ticker
```

This exports training CSVs (**data ≤ 2025-07-01 only** — the OOS year is never
shown to training), downloads the pretrained weights locally, and writes one
config per ticker for a **predictor-only** fine-tune (the tokenizer stays
frozen: ~1.4k daily bars is too little to retrain a quantizer).

**Launch** (from `finetune_csv\`, one per ticker, ~10–20 min each on the 5080):

```powershell
cd finetune_csv
..\.venv\Scripts\python.exe train_sequential.py --config configs/config_CW8_PA_1d.yaml --skip-tokenizer
```

**Evaluate** — IC first, PnL second:

1. Point `model.name` in `trading/config.yaml` at
   `finetune_csv/finetuned/{TICKER}_1d/basemodel/best_model` (see the comment
   in the config; sweep caches are tagged per model so runs never mix).
2. `python -m trading.sweep --end 2025-07-01 --tickers <TICKER>` then
   `python -m trading.ic_report` — **success gate: IC above the 2/√n noise
   bar where zero-shot was ≈ 0.** If IC doesn't move, stop there.
3. Only if the IC gate passes:
   `python -m trading.backtest --oos-start 2025-07-01 --tickers <TICKER>` and
   judge the OOS block against buy-and-hold *and* momentum, as always.

Tweakables in `trading/export_finetune.py` flags: `--epochs` (20),
`--batch-size` (16), `--lr` (2e-6 — deliberately gentle; raise cautiously,
catastrophic forgetting on 1.4k bars is the main risk).

---

## Caveats (read once)
- **Zero-shot on unfamiliar markets is unproven.** Kronos was trained largely on
  other exchanges; the backtest is exactly how you find out if it transfers.
- **Forecasts are stochastic** — raising `sample_count` averages out noise but
  costs time.
- **Costs & spreads are real.** `fee_bps` is a rough proxy; EM names have wider
  spreads than the config default assumes.
- **Thin/illiquid tickers** (some Africa/Asia names) trigger data-quality
  warnings; treat their results with extra skepticism.
- This is a **research tool, not financial advice.** No live money until
  out-of-sample results are convincing.
