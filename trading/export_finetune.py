"""Set up per-ticker fine-tuning experiments for the finetune_csv pipeline.

Fine-tuning is the last untried edge lever (two zero-shot campaigns failed the
OOS gate). This script prepares everything so a run is ONE command later:

  * exports each ticker's daily bars to finetune_csv/data/{TICKER}_1d.csv —
    ONLY through --end (default 2025-07-01), so the OOS confirmation year
    stays untouched by training;
  * downloads the pretrained model + tokenizer to finetune_csv/pretrained/
    (the pipeline needs local directories, not HF hub ids);
  * writes finetune_csv/configs/config_{TICKER}_1d.yaml for a
    predictor-only fine-tune (tokenizer stays frozen at the pretrained one:
    ~1.4k daily bars is far too little to retrain a quantizer).

One CSV per config on purpose: the pipeline slides windows over a single
contiguous series, so concatenating tickers would create corrupt windows that
span two instruments.

Launch (later, from finetune_csv\\, one per ticker, ~10-20 min each on the 5080):
    ..\\.venv\\Scripts\\python.exe train_sequential.py --config configs/config_CW8_PA_1d.yaml --skip-tokenizer

Evaluate (success gate BEFORE any PnL talk: IC above the noise bar):
    1. point trading/config.yaml model.name at
       finetune_csv/finetuned/{TICKER}_1d/basemodel/best_model
    2. python -m trading.sweep --end 2025-07-01 --tickers <TICKER>
    3. python -m trading.ic_report
    4. python -m trading.backtest --oos-start 2025-07-01 --tickers <TICKER>
"""
from __future__ import annotations

import argparse
import copy
import os

import yaml

from trading.config import load_config
from trading.data import load_ticker

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FT = os.path.join(_REPO, "finetune_csv")

DEFAULT_TICKERS = ["CW8.PA", "EUNM.DE", "AAPL"]


def _safe(ticker: str) -> str:
    return ticker.replace("/", "_").replace(".", "_")


def ensure_pretrained(model_id: str) -> str:
    """Snapshot a HF model into finetune_csv/pretrained/<name> (once)."""
    dest = os.path.join(_FT, "pretrained", model_id.split("/")[-1])
    if os.path.exists(os.path.join(dest, "config.json")):
        return dest
    from huggingface_hub import snapshot_download
    print(f"Downloading {model_id} -> {dest} ...")
    snapshot_download(repo_id=model_id, local_dir=dest)
    return dest


def export_csv(ticker: str, cfg: dict, end: str) -> tuple[str, int]:
    """Write the training CSV (bars strictly <= `end`) and return (path, n)."""
    cfg = copy.deepcopy(cfg)
    cfg["data"]["history_end"] = end
    df = load_ticker(ticker, cfg)
    path = os.path.join(_FT, "data", f"{_safe(ticker)}_1d.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path, len(df)


_VAL_RATIO = 0.20


def _window_sizes(n_bars: int) -> tuple[int, int]:
    """(lookback_window, predict_window) sized to the data.

    Constraints: window = lookback + predict + 1 must (a) stay <= max_context
    (512) because the trainer feeds the whole window through the model, and
    (b) fit inside the chronological VALIDATION split with headroom — daily
    histories (~1.4k bars) are far shorter than the 5-minute template's, and
    a window larger than the val split crashes the pipeline with
    "Available samples: -370".
    """
    window = max(128, min(512, int(n_bars * _VAL_RATIO) - 40))
    predict = max(8, window // 8)
    return window - predict - 1, predict


def write_experiment_config(ticker: str, csv_path: str, n_bars: int,
                            tok_dir: str, pred_dir: str,
                            epochs: int, batch_size: int, lr: float, end: str) -> str:
    exp_name = f"{_safe(ticker)}_1d"
    lookback_window, predict_window = _window_sizes(n_bars)
    config = {
        "data": {
            "data_path": csv_path,
            "lookback_window": lookback_window,
            "predict_window": predict_window,
            "max_context": 512,
            "clip": 5.0,
            # chronological split inside the <= --end data; the post-`end`
            # OOS year is absent from this CSV entirely.
            "train_ratio": 1.0 - _VAL_RATIO,
            "val_ratio": _VAL_RATIO,
            "test_ratio": 0.0,
        },
        "training": {
            "tokenizer_epochs": 1,        # unused: tokenizer training is disabled
            "basemodel_epochs": epochs,
            "batch_size": batch_size,
            "log_interval": 20,
            "num_workers": 0,             # Windows: >0 spawns broken workers
            "seed": 42,
            "tokenizer_learning_rate": 2.0e-4,
            "predictor_learning_rate": lr,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_weight_decay": 0.1,
            "accumulation_steps": 1,
        },
        "model_paths": {
            "pretrained_tokenizer": tok_dir,
            "pretrained_predictor": pred_dir,
            "exp_name": exp_name,
            "base_path": os.path.join(_FT, "finetuned"),
            "base_save_path": "",         # auto: {base_path}/{exp_name}
            # predictor-only fine-tune: the "fine-tuned" tokenizer IS the
            # pretrained one (frozen) — the trainer loads it from this path.
            "finetuned_tokenizer": tok_dir,
            "tokenizer_save_name": "tokenizer",
            "basemodel_save_name": "basemodel",
        },
        "experiment": {
            "name": f"kronos_{exp_name}",
            "description": f"Predictor-only daily fine-tune on {ticker} "
                           f"(data <= {end}; OOS year untouched)",
            "use_comet": False,
            "train_tokenizer": False,     # frozen pretrained tokenizer
            "train_basemodel": True,
            "skip_existing": False,
        },
        "device": {"use_cuda": True, "device_id": 0},
    }
    path = os.path.join(_FT, "configs", f"config_{exp_name}.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, default_flow_style=False, sort_keys=False)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare per-ticker Kronos fine-tuning experiments")
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--end", default="2025-07-01",
                    help="training data cutoff — keep it at the OOS split date")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-6,
                    help="predictor learning rate (template used 1e-6 on much more data)")
    ap.add_argument("--config", default=None, help="trading config path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok_dir = ensure_pretrained(cfg["model"]["tokenizer"])
    pred_dir = ensure_pretrained(cfg["model"]["name"])

    launches = []
    for ticker in args.tickers:
        try:
            csv_path, n = export_csv(ticker, cfg, args.end)
        except Exception as e:
            print(f"SKIPPED {ticker}: {e}")
            continue
        yaml_path = write_experiment_config(ticker, csv_path, n, tok_dir, pred_dir,
                                            args.epochs, args.batch_size, args.lr, args.end)
        lb, pw = _window_sizes(n)
        window = lb + pw + 1
        n_train = max(0, int(n * (1.0 - _VAL_RATIO)) - window + 1)
        n_val = max(0, n - int(n * (1.0 - _VAL_RATIO)) - window + 1)
        print(f"{ticker}: {n} bars <= {args.end} -> {os.path.relpath(csv_path, _REPO)} "
              f"(window {window} = {lb}+{pw}+1; {n_train} train / {n_val} val windows); "
              f"config -> {os.path.relpath(yaml_path, _REPO)}")
        launches.append(os.path.basename(yaml_path))

    if launches:
        print("\nLaunch when ready (from finetune_csv\\, one per ticker):")
        for name in launches:
            print(f"  ..\\.venv\\Scripts\\python.exe train_sequential.py "
                  f"--config configs/{name} --skip-tokenizer")
        print("\nThen evaluate per trading/README.md section 5 "
              "(IC above the noise bar first, PnL second).")


if __name__ == "__main__":
    main()
