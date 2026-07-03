import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_SIGNAL_PAIRS = {
    2: ("smcAC", "vib_spindle"),
    11: ("smcAC", "vib_spindle"),
}
PAPER_SEGMENTS_PER_RUN = {
    2: 8,
    11: 5,
}


@dataclass
class SegmentSample:
    case_id: int
    run_id: int
    segment_id: int
    features: np.ndarray
    vb: float


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    target_mean: float
    target_std: float

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.target_mean) / self.target_std).astype(np.float32)

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.target_std + self.target_mean


class WearDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_signal(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values).reshape(-1).astype(np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float64)

    valid = x[finite]
    valid = valid[np.abs(valid) < 1e6]
    if valid.size == 0:
        valid = x[finite]

    low, high = np.percentile(valid, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        median = float(np.median(valid))
        mad = float(np.median(np.abs(valid - median)))
        width = max(6.0 * mad, 1e-6)
        low, high = median - width, median + width

    fill = float(np.median(valid))
    x = np.where(np.abs(x) < 1e6, x, np.nan)
    x = np.nan_to_num(x, nan=fill, posinf=high, neginf=low)
    return np.clip(x, low, high)


def dwt_soft_threshold(signal: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    x = clean_signal(signal)
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    use_level = max(1, min(level, max_level))
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=use_level, mode="symmetric")
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if detail.size else 0.0
    threshold = sigma * math.sqrt(2.0 * math.log(max(len(x), 2)))
    filtered = [coeffs[0]]
    filtered.extend(pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:])
    y = pywt.waverec(filtered, wavelet=wavelet, mode="symmetric")[: len(x)]
    return clean_signal(y)


def lowess_smooth(values: np.ndarray, frac: float = 0.2) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    n = y.size
    if n < 4:
        return y.astype(np.float32)

    x = np.arange(n, dtype=np.float64)
    bandwidth = max(2, int(math.ceil(frac * n)))
    smoothed = np.empty_like(y)
    for i in range(n):
        distances = np.abs(x - x[i])
        nearest = np.argpartition(distances, bandwidth - 1)[:bandwidth]
        max_dist = max(float(distances[nearest].max()), 1e-12)
        u = distances[nearest] / max_dist
        weights = (1.0 - u**3) ** 3
        design = np.column_stack([np.ones_like(nearest, dtype=np.float64), x[nearest] - x[i]])
        w_design = design * weights[:, None]
        try:
            beta = np.linalg.pinv(w_design.T @ design) @ (w_design.T @ y[nearest])
            smoothed[i] = beta[0]
        except np.linalg.LinAlgError:
            smoothed[i] = np.average(y[nearest], weights=weights)
    return smoothed.astype(np.float32)


def safe_div(num: float, den: float, eps: float = 1e-12) -> float:
    return float(num / (den + eps))


def twelve_features(signal: np.ndarray) -> np.ndarray:
    x = clean_signal(signal)
    n = max(x.size, 1)
    mean = float(np.mean(x))
    centered = x - mean
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x**2)))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    peak = float(np.max(abs_x))
    ptp = float(np.ptp(x))
    abs_mean = float(np.mean(abs_x))
    sqrt_abs_mean = float(np.mean(np.sqrt(abs_x + 1e-12)))
    skew = float(np.mean(centered**3) / (std**3 + 1e-12)) if std > 0 else 0.0
    kurt = float(np.mean(centered**4) / (std**4 + 1e-12)) if std > 0 else 0.0
    crest = safe_div(peak, rms)
    clearance = safe_div(peak, sqrt_abs_mean**2)
    form = safe_div(n * rms, np.sum(abs_x))

    spectrum = np.fft.rfft(x)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0)
    power_sum = float(np.sum(power))
    center_freq = safe_div(float(np.sum(freqs * power)), power_sum)
    rms_freq = math.sqrt(max(safe_div(float(np.sum((freqs**2) * power)), power_sum), 0.0))
    freq_std = math.sqrt(max(safe_div(float(np.sum(((freqs - center_freq) ** 2) * power)), power_sum), 0.0))

    features = np.asarray(
        [
            mean,
            ptp,
            rms,
            std,
            skew,
            kurt,
            crest,
            clearance,
            form,
            center_freq,
            rms_freq,
            freq_std,
        ],
        dtype=np.float64,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(features, -1e6, 1e6).astype(np.float32)


def scalar_field(record, name: str, default: float = np.nan) -> float:
    if name not in record.dtype.names:
        return default
    value = np.asarray(record[name]).squeeze()
    if value.size == 0:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_case_records(data_root: Path, mat_file: str, case_id: int) -> List:
    records = loadmat(data_root / mat_file)["mill"][0]
    selected = [record for record in records if int(scalar_field(record, "case", 0)) == case_id]
    selected.sort(key=lambda record: int(scalar_field(record, "run", 0)))
    return [record for record in selected if np.isfinite(scalar_field(record, "VB"))]


def interpolate_segment_vb(vbs: np.ndarray, segments_per_run: int) -> np.ndarray:
    known_x = np.arange(segments_per_run - 1, len(vbs) * segments_per_run, segments_per_run, dtype=np.float64)
    all_x = np.arange(len(vbs) * segments_per_run, dtype=np.float64)
    if len(vbs) < 4:
        labels = np.interp(all_x, known_x, vbs)
    else:
        spline = CubicSpline(known_x, vbs, bc_type="natural", extrapolate=True)
        labels = spline(all_x)

    labels = np.clip(labels, 0.0, None)
    for run_index, vb in enumerate(vbs):
        end = (run_index + 1) * segments_per_run
        start = end - segments_per_run
        labels[end - 1] = vb
        lower_bound = vbs[run_index - 1] if run_index > 0 else min(vb, labels[start])
        labels[start:end] = np.clip(labels[start:end], min(lower_bound, vb), max(lower_bound, vb))
    return labels.astype(np.float32)


def segment_indices(length: int, n_segments: int) -> Iterable[Tuple[int, int]]:
    points = np.linspace(0, length, n_segments + 1, dtype=np.int64)
    for start, end in zip(points[:-1], points[1:]):
        yield int(start), int(max(start + 1, end))


def build_case_samples(
    data_root: Path,
    mat_file: str,
    case_id: int,
    current_channel: str,
    vibration_channel: str,
    wavelet: str,
    wavelet_level: int,
    lowess_frac: float,
) -> List[SegmentSample]:
    records = load_case_records(data_root, mat_file, case_id)
    segments_per_run = PAPER_SEGMENTS_PER_RUN[case_id]
    vbs = np.asarray([scalar_field(record, "VB") for record in records], dtype=np.float32)
    labels = interpolate_segment_vb(vbs, segments_per_run)
    samples: List[SegmentSample] = []
    label_index = 0

    for record in records:
        run_id = int(scalar_field(record, "run", 0))
        current = dwt_soft_threshold(record[current_channel], wavelet=wavelet, level=wavelet_level)
        vibration = dwt_soft_threshold(record[vibration_channel], wavelet=wavelet, level=wavelet_level)
        signal_length = min(len(current), len(vibration))
        current = current[:signal_length]
        vibration = vibration[:signal_length]

        run_features = []
        for start, end in segment_indices(signal_length, segments_per_run):
            features = np.concatenate(
                [
                    twelve_features(current[start:end]),
                    twelve_features(vibration[start:end]),
                ],
                axis=0,
            )
            run_features.append(features)
        run_features = np.stack(run_features, axis=0)

        for feature_idx in range(run_features.shape[1]):
            run_features[:, feature_idx] = lowess_smooth(run_features[:, feature_idx], frac=lowess_frac)

        for segment_id, features in enumerate(run_features, start=1):
            samples.append(
                SegmentSample(
                    case_id=case_id,
                    run_id=run_id,
                    segment_id=segment_id,
                    features=features.astype(np.float32),
                    vb=float(labels[label_index]),
                )
            )
            label_index += 1

    return samples


def make_sequences(samples: Sequence[SegmentSample], sequence_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.stack([sample.features for sample in samples], axis=0)
    targets = np.asarray([sample.vb for sample in samples], dtype=np.float32)
    meta = np.asarray([(sample.case_id, sample.run_id, sample.segment_id) for sample in samples], dtype=np.int64)
    x_rows = []
    y_rows = []
    meta_rows = []
    for end in range(sequence_length - 1, len(samples)):
        start = end - sequence_length + 1
        x_rows.append(features[start : end + 1])
        y_rows.append(targets[end])
        meta_rows.append(meta[end])
    return (
        np.stack(x_rows, axis=0).astype(np.float32),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(meta_rows, dtype=np.int64),
    )


def random_train_test_split(n_samples: int, train_ratio: float, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.arange(n_samples)
    rng.shuffle(order)
    train_count = max(1, min(n_samples - 1, int(round(n_samples * train_ratio))))
    return {
        "train": order[:train_count],
        "test": order[train_count:],
    }


def fit_normalizer(x_train: np.ndarray, y_train: np.ndarray) -> Normalizer:
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), 1e-6)
    return Normalizer(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        target_mean=float(y_train.mean()),
        target_std=float(max(y_train.std(), 1e-6)),
    )


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x), inplace=True)


class ResidualAttentionBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int, heads: int, dropout: float):
        super().__init__()
        self.residual = ResidualBlock1D(channels, kernel_size=kernel_size, dropout=dropout)
        self.attention = nn.MultiheadAttention(channels, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.residual(x)
        sequence = local.transpose(1, 2)
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        return self.norm(sequence + attended).transpose(1, 2)


class PMSRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        conv_channels: int = 48,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList(
            [
                ResidualAttentionBranch(conv_channels, kernel_size=3, heads=heads, dropout=dropout),
                ResidualAttentionBranch(conv_channels, kernel_size=5, heads=heads, dropout=dropout),
                ResidualAttentionBranch(conv_channels, kernel_size=7, heads=heads, dropout=dropout),
            ]
        )
        self.global_branch = nn.Sequential(
            nn.Conv1d(conv_channels, conv_channels, kernel_size=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(conv_channels * 4, conv_channels, kernel_size=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.temporal_score = nn.Linear(lstm_hidden * 2, 1, bias=False)
        self.regressor = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        stem = self.stem(x)
        branches = [branch(stem) for branch in self.branches]
        branches.append(self.global_branch(stem))
        fused = self.fusion(torch.cat(branches, dim=1)).transpose(1, 2)
        encoded, _ = self.encoder(fused)
        weights = torch.softmax(self.temporal_score(encoded), dim=1)
        context = torch.sum(encoded * weights, dim=1)
        return self.regressor(context)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    nonzero = np.abs(y_true) > 1e-12
    mape = float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100.0) if np.any(nonzero) else 0.0
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
        "MAE_um": mae * 1000.0,
        "RMSE_um": rmse * 1000.0,
    }


def predict(model: nn.Module, loader: DataLoader, normalizer: Normalizer, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy().reshape(-1)
            preds.append(pred)
            targets.append(y.numpy().reshape(-1))
    pred_norm = np.concatenate(preds)
    true_norm = np.concatenate(targets)
    return normalizer.inverse_y(true_norm), normalizer.inverse_y(pred_norm)


def train_case(args, case_id: int, device: torch.device) -> Dict:
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

    model = PMSRegressor(
        input_dim=x.shape[-1],
        conv_channels=args.conv_channels,
        lstm_hidden=args.hidden_dim,
        lstm_layers=2,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_period, gamma=args.lr_decay_factor)
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
                f"Case {case_id} epoch={epoch} iter={iteration} "
                f"test_RMSE={test_metrics['RMSE_um']:.2f}um "
                f"test_MAE={test_metrics['MAE_um']:.2f}um"
            )

    train_true, train_pred = predict(model, loaders["train"], normalizer, device)
    test_true, test_pred = predict(model, loaders["test"], normalizer, device)
    train_metrics = regression_metrics(train_true, train_pred)
    test_metrics = regression_metrics(test_true, test_pred)

    case_dir = Path(args.output_dir) / f"case_{case_id}"
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
        "channels": {
            "current": current_channel,
            "vibration": vibration_channel,
        },
        "segments_per_run": PAPER_SEGMENTS_PER_RUN[case_id],
        "n_valid_runs": len(load_case_records(Path(args.data_root), args.mat_file, case_id)),
        "n_augmented_segments": len(samples),
        "sequence_length": args.sequence_length,
        "n_sequences": len(y),
        "split_sizes": {name: int(len(index)) for name, index in split.items()},
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def write_result_text(path: Path, summary: Dict) -> None:
    lines = [
        "Paper-like PMS experiment results",
        "=================================",
        "",
        f"Device: {summary['device']}",
        f"Max iterations: {summary['max_iterations']}",
        f"Learning rate: {summary['lr']}",
        f"Batch size: {summary['batch_size']}",
        f"Train ratio: {summary['train_ratio']}",
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
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-like PMS few-sample tool-wear experiment.")
    parser.add_argument("--data-root", default="3. Milling")
    parser.add_argument("--mat-file", default="mill.mat")
    parser.add_argument("--output-dir", default=str(CURRENT_DIR / "outputs" / "paper_like_pms"))
    parser.add_argument("--cases", type=int, nargs="+", default=[11, 2])
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.012)
    parser.add_argument("--lr-decay-factor", type=float, default=0.892)
    parser.add_argument("--lr-decay-period", type=int, default=1200)
    parser.add_argument("--conv-channels", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    cases = []
    for case_id in args.cases:
        if case_id not in PAPER_SEGMENTS_PER_RUN:
            raise ValueError(f"Only paper cases are supported: {sorted(PAPER_SEGMENTS_PER_RUN)}")
        cases.append(train_case(args, case_id=case_id, device=device))

    summary = {
        "device": str(device),
        "data_root": args.data_root,
        "mat_file": args.mat_file,
        "model": "PRes-MHSA-SBiLSTM paper-like regressor",
        "preprocessing": {
            "dwt": True,
            "emd": "not applied; no EMD package is installed in the current environment",
            "lowess": "local weighted smoothing fallback",
            "vb_interpolation": "cubic spline over augmented segments",
        },
        "features": "12 features from current + 12 features from vibration = 24 features",
        "train_ratio": args.train_ratio,
        "batch_size": args.batch_size,
        "max_iterations": args.max_iterations,
        "lr": args.lr,
        "lr_decay_factor": args.lr_decay_factor,
        "lr_decay_period": args.lr_decay_period,
        "cases": cases,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    write_result_text(CURRENT_DIR / "paper_like_results.txt", summary)

    print("\nFinal paper-like metrics")
    print(json.dumps({case["case_id"]: case["test_metrics"] for case in cases}, indent=2))
    print(f"\nSaved outputs to: {output_dir}")
    print(f"Recorded text results to: {CURRENT_DIR / 'paper_like_results.txt'}")


if __name__ == "__main__":
    main()
