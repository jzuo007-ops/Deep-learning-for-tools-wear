import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stacked_bilstm_reproduction.paper_like_pms_experiment import (
    DEFAULT_SIGNAL_PAIRS,
    PAPER_SEGMENTS_PER_RUN,
    WearDataset,
    build_case_samples,
    fit_normalizer,
    load_case_records,
    make_sequences,
    predict,
    random_train_test_split,
    regression_metrics,
    set_seed,
)


CURRENT_DIR = Path(__file__).resolve().parent


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = torch.tanh(self.proj(x))
        weights = torch.softmax(self.score(energy), dim=1)
        return torch.sum(weights * x, dim=1)


class CNNBiLSTMAttentionRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        conv_channels: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        attention_dim: int = 64,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attention = TemporalAttention(hidden_dim * 2, attention_dim=attention_dim)
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        local = self.cnn(x).transpose(1, 2)
        encoded, _ = self.encoder(local)
        context = self.attention(encoded)
        return self.regressor(context)


def train_case(args, case_id: int, device: torch.device):
    current_channel, vibration_channel = DEFAULT_SIGNAL_PAIRS[case_id]
    samples = build_case_samples(
        data_root=Path(args.data_root),
        mat_file=args.mat_file,
        case_id=case_id,
        current_channel=current_channel,
        vibration_channel=vibration_channel,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        lowess_frac=args.lowess_frac,
    )
    x, y, meta = make_sequences(samples, sequence_length=args.sequence_length)
    split = random_train_test_split(len(y), train_ratio=args.train_ratio, seed=args.seed + case_id)
    normalizer = fit_normalizer(x[split["train"]], y[split["train"]])
    x_norm = normalizer.transform_x(x)
    y_norm = normalizer.transform_y(y)

    loaders = {
        name: DataLoader(
            WearDataset(x_norm[index], y_norm[index]),
            batch_size=args.batch_size,
            shuffle=name == "train",
        )
        for name, index in split.items()
    }

    model = CNNBiLSTMAttentionRegressor(
        input_dim=x.shape[-1],
        conv_channels=args.conv_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        attention_dim=args.attention_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_period,
        gamma=args.lr_decay_factor,
    )
    loss_fn = nn.MSELoss()
    log_rows = []

    iteration = 0
    epoch = 0
    while iteration < args.max_iterations:
        epoch += 1
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y in loaders["train"]:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * batch_x.shape[0]
            seen += batch_x.shape[0]
            iteration += 1
            if iteration >= args.max_iterations:
                break

        if epoch == 1 or iteration >= args.max_iterations or epoch % args.log_every == 0:
            train_true, train_pred = predict(model, loaders["train"], normalizer, device)
            test_true, test_pred = predict(model, loaders["test"], normalizer, device)
            train_metrics = regression_metrics(train_true, train_pred)
            test_metrics = regression_metrics(test_true, test_pred)
            row = {
                "epoch": epoch,
                "iteration": iteration,
                "train_loss": total_loss / max(seen, 1),
                "train_RMSE_um": train_metrics["RMSE_um"],
                "test_RMSE_um": test_metrics["RMSE_um"],
                "test_MAE_um": test_metrics["MAE_um"],
                "test_R2": test_metrics["R2"],
            }
            log_rows.append(row)
            print(
                f"CNN-BiLSTM-Attention case={case_id} epoch={epoch} iter={iteration} "
                f"test_RMSE={test_metrics['RMSE_um']:.2f}um "
                f"test_MAE={test_metrics['MAE_um']:.2f}um"
            )

    train_true, train_pred = predict(model, loaders["train"], normalizer, device)
    test_true, test_pred = predict(model, loaders["test"], normalizer, device)
    train_metrics = regression_metrics(train_true, train_pred)
    test_metrics = regression_metrics(test_true, test_pred)

    case_dir = CURRENT_DIR / f"case_{case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    with (case_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["case_id", "run_id", "segment_id", "true_VB_mm", "predicted_VB_mm", "absolute_error_mm"])
        for meta_row, true_value, pred_value in zip(meta[split["test"]], test_true, test_pred):
            writer.writerow(
                [
                    int(meta_row[0]),
                    int(meta_row[1]),
                    int(meta_row[2]),
                    float(true_value),
                    float(pred_value),
                    abs(float(pred_value - true_value)),
                ]
            )

    return {
        "case_id": case_id,
        "paper_experiment": "E1" if case_id == 11 else "E2",
        "segments_per_run": PAPER_SEGMENTS_PER_RUN[case_id],
        "n_valid_runs": len(load_case_records(Path(args.data_root), args.mat_file, case_id)),
        "n_augmented_segments": len(samples),
        "sequence_length": args.sequence_length,
        "n_sequences": len(y),
        "split_sizes": {name: int(len(index)) for name, index in split.items()},
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def write_results(summary):
    lines = [
        "CNN-BiLSTM-Attention VB regression experiment",
        "=============================================",
        "",
        f"Device: {summary['device']}",
        f"Train ratio: {summary['train_ratio']}",
        f"Batch size: {summary['batch_size']}",
        f"Max iterations: {summary['max_iterations']}",
        f"Learning rate: {summary['lr']}",
        "",
    ]
    for case in summary["cases"]:
        metrics = case["test_metrics"]
        lines.extend(
            [
                f"Case {case['case_id']} / {case['paper_experiment']}",
                f"- Valid runs: {case['n_valid_runs']}",
                f"- Augmented segments: {case['n_augmented_segments']}",
                f"- Sequences: {case['n_sequences']}",
                f"- Split sizes: {case['split_sizes']}",
                f"- MAE: {metrics['MAE']:.6f} mm ({metrics['MAE_um']:.2f} um)",
                f"- RMSE: {metrics['RMSE']:.6f} mm ({metrics['RMSE_um']:.2f} um)",
                f"- R2: {metrics['R2']:.6f}",
                f"- MAPE: {metrics['MAPE_percent']:.2f}%",
                "",
            ]
        )
    (CURRENT_DIR / "results.txt").write_text("\n".join(lines), encoding="utf-8")
    with (CURRENT_DIR / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="CNN-BiLSTM-Attention VB regression experiment.")
    parser.add_argument("--data-root", default=str(ROOT / "3. Milling"))
    parser.add_argument("--mat-file", default="mill.mat")
    parser.add_argument("--cases", type=int, nargs="+", default=[11, 2])
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.012)
    parser.add_argument("--lr-decay-factor", type=float, default=0.892)
    parser.add_argument("--lr-decay-period", type=int, default=1200)
    parser.add_argument("--conv-channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--wavelet-level", type=int, default=3)
    parser.add_argument("--lowess-frac", type=float, default=0.35)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    cases = [train_case(args, case_id=case_id, device=device) for case_id in args.cases]
    summary = {
        "model": "CNN-BiLSTM-Attention",
        "device": str(device),
        "data_root": args.data_root,
        "preprocessing": "DWT + cubic VB interpolation + LOWESS-style smoothing",
        "features": "12 current features + 12 vibration features",
        "train_ratio": args.train_ratio,
        "batch_size": args.batch_size,
        "max_iterations": args.max_iterations,
        "lr": args.lr,
        "cases": cases,
    }
    write_results(summary)
    print(json.dumps({case["case_id"]: case["test_metrics"] for case in cases}, indent=2))
    print(f"Saved results to {CURRENT_DIR}")


if __name__ == "__main__":
    main()
