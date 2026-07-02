import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dataset import WearSequenceDataset, apply_normalizer, build_sequence_samples, fit_normalizer, make_split
from feature_extraction import DEFAULT_CHANNELS, load_milling_records
from metrics import regression_metrics
from model import StackedBiLSTMAttentionRegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Few-sample stacked BiLSTM tool-wear prediction.")
    parser.add_argument("--data-root", default="3. Milling", help="Folder containing mill.mat.")
    parser.add_argument("--mat-file", default="mill.mat")
    parser.add_argument("--output-dir", default=str(CURRENT_DIR / "outputs" / "single_run"))
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--predict-next", action="store_true", help="Predict the next run VB instead of the last run in the window.")
    parser.add_argument("--split-mode", choices=["chronological", "case_holdout", "random"], default="chronological")
    parser.add_argument("--train-ratio", type=float, default=0.30)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def make_loaders(args):
    runs = load_milling_records(args.data_root, mat_file=args.mat_file, channels=DEFAULT_CHANNELS)
    x, y, meta = build_sequence_samples(runs, lookback=args.lookback, predict_next=args.predict_next)
    split = make_split(
        x,
        y,
        meta,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if len(split["train"]) == 0 or len(split["val"]) == 0 or len(split["test"]) == 0:
        raise ValueError(f"Empty split produced: { {key: len(value) for key, value in split.items()} }")

    normalizer = fit_normalizer(x[split["train"]], y[split["train"]])
    x_norm, y_norm = apply_normalizer(x, y, normalizer)

    datasets = {
        name: WearSequenceDataset(x_norm[index], y_norm[index])
        for name, index in split.items()
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
    }
    raw_targets = {name: y[index] for name, index in split.items()}
    raw_meta = {name: meta[index] for name, index in split.items()}
    return loaders, normalizer, raw_targets, raw_meta, x.shape[-1], len(runs), len(y)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        prediction, _ = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        running_loss += loss.item() * x.shape[0]
    return running_loss / max(len(loader.dataset), 1)


def predict(model, loader, device, normalizer):
    model.eval()
    predictions = []
    targets = []
    attention = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prediction, weights = model(x)
            predictions.append(prediction.cpu().numpy().reshape(-1))
            targets.append(y.numpy().reshape(-1))
            attention.append(weights.cpu().numpy())
    y_pred_norm = np.concatenate(predictions)
    y_true_norm = np.concatenate(targets)
    y_pred = normalizer.inverse_target(y_pred_norm)
    y_true = normalizer.inverse_target(y_true_norm)
    return y_true, y_pred, np.concatenate(attention, axis=0)


def evaluate(model, loader, device, normalizer):
    y_true, y_pred, attention = predict(model, loader, device, normalizer)
    return regression_metrics(y_true, y_pred), y_true, y_pred, attention


def write_predictions(path: Path, meta: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["case_id", "run_id", "true_VB", "predicted_VB", "absolute_error"])
        for (case_id, run_id), true_value, pred_value in zip(meta, y_true, y_pred):
            writer.writerow([int(case_id), int(run_id), float(true_value), float(pred_value), abs(float(pred_value - true_value))])


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    loaders, normalizer, raw_targets, raw_meta, input_dim, n_runs, n_sequences = make_loaders(args)

    model = StackedBiLSTMAttentionRegressor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        attention_dim=args.attention_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=15, factor=0.5)
    loss_fn = nn.SmoothL1Loss()

    best_val_rmse = float("inf")
    best_state = None
    stale_epochs = 0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, loss_fn, device)
        val_metrics, _, _, _ = evaluate(model, loaders["val"], device, normalizer)
        scheduler.step(val_metrics["RMSE"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        log_rows.append(row)
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"loss={train_loss:.6f} val_RMSE={val_metrics['RMSE']:.6f} "
            f"val_MAE={val_metrics['MAE']:.6f} val_R2={val_metrics['R2']:.6f}"
        )

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics, train_true, train_pred, _ = evaluate(model, loaders["train"], device, normalizer)
    val_metrics, val_true, val_pred, _ = evaluate(model, loaders["val"], device, normalizer)
    test_metrics, test_true, test_pred, attention = evaluate(model, loaders["test"], device, normalizer)

    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(log_rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    write_predictions(output_dir / "test_predictions.csv", raw_meta["test"], test_true, test_pred)
    np.save(output_dir / "test_attention.npy", attention)

    summary = {
        "data_root": args.data_root,
        "mat_file": args.mat_file,
        "device": str(device),
        "n_runs": n_runs,
        "n_sequences": n_sequences,
        "split_mode": args.split_mode,
        "lookback": args.lookback,
        "predict_next": args.predict_next,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "split_sizes": {key: len(value.dataset) for key, value in loaders.items()},
        "model": {
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "attention_dim": args.attention_dim,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nFinal metrics")
    print(json.dumps(summary["test_metrics"], indent=2))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()

