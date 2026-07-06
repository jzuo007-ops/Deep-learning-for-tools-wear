import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phm2010_segmentation.dataset import normalize_window
from phm2010_segmentation.label_cache import cache_path_for_cut, load_label_cache
from phm2010_segmentation.pseudo_label import (
    CHANNEL_NAMES,
    CLASS_COLORS,
    CLASS_NAMES,
    PseudoLabelConfig,
    generate_three_class_labels,
)
from src.deeplabv3_model import DeepLabV3_1D


CURRENT_DIR = Path(__file__).resolve().parent


def load_prediction(checkpoint: Path, data: np.ndarray, crop_length: int, device: torch.device, backbone: str):
    model = DeepLabV3_1D(
        in_channels=7,
        num_classes=3,
        aux_loss=True,
        classification=False,
        backbone_name=backbone,
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    preds = np.zeros(len(data), dtype=np.int64)
    counts = np.zeros(len(data), dtype=np.float32)
    stride = crop_length // 2
    starts = list(range(0, max(1, len(data) - crop_length + 1), stride))
    if starts[-1] != max(0, len(data) - crop_length):
        starts.append(max(0, len(data) - crop_length))
    with torch.no_grad():
        for start in starts:
            end = min(start + crop_length, len(data))
            window = data[start:end]
            if len(window) < crop_length:
                window = np.pad(window, ((0, crop_length - len(window)), (0, 0)), mode="edge")
            signal = torch.from_numpy(normalize_window(window).T).float().unsqueeze(0).to(device)
            out = model(signal)["out"].argmax(dim=1).cpu().numpy()[0][: end - start]
            preds[start:end] += out
            counts[start:end] += 1
    return np.rint(preds / np.maximum(counts, 1.0)).astype(np.int64)


def add_label_spans(ax, labels: np.ndarray, x: np.ndarray, alpha: float = 0.22):
    labels = np.asarray(labels)
    change_points = np.where(np.diff(labels) != 0)[0] + 1
    starts = np.concatenate([[0], change_points])
    ends = np.concatenate([change_points, [len(labels)]])
    for start, end in zip(starts, ends):
        label = int(labels[start])
        ax.axvspan(x[start], x[end - 1], color=CLASS_COLORS[label], alpha=alpha, linewidth=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PHM 2010 segmentation pseudo labels or predictions.")
    parser.add_argument("--cut-file", default=str(ROOT / "PHM 2010" / "c1" / "c_1_253.csv"))
    parser.add_argument("--output", default=str(CURRENT_DIR / "outputs" / "pseudo_label_example.png"))
    parser.add_argument("--data-root", default=str(ROOT / "PHM 2010"))
    parser.add_argument("--label-cache-dir", default=str(CURRENT_DIR / "label_cache"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "lstm"])
    parser.add_argument("--crop-length", type=int, default=8192)
    parser.add_argument("--smooth-window", type=int, default=2048)
    parser.add_argument("--active-threshold", type=float, default=0.25)
    parser.add_argument("--inactive-threshold", type=float, default=0.12)
    parser.add_argument("--transition-ratio", type=float, default=0.05)
    parser.add_argument("--min-transition-points", type=int, default=4096)
    parser.add_argument("--min-active-points", type=int, default=8192)
    parser.add_argument("--min-cut-ratio", type=float, default=0.35)
    parser.add_argument("--max-gap-ratio", type=float, default=0.03)
    parser.add_argument("--max-gap-points", type=int, default=8192)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.01)
    parser.add_argument("--allow-label-cache-config-mismatch", action="store_true")
    parser.add_argument("--max-plot-points", type=int, default=30000)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.cut_file)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
    config = PseudoLabelConfig(
        smooth_window=args.smooth_window,
        active_threshold=args.active_threshold,
        inactive_threshold=args.inactive_threshold,
        transition_ratio=args.transition_ratio,
        min_transition_points=args.min_transition_points,
        min_active_points=args.min_active_points,
        min_cut_ratio=args.min_cut_ratio,
        max_gap_ratio=args.max_gap_ratio,
        max_gap_points=args.max_gap_points,
        edge_margin_ratio=args.edge_margin_ratio,
    )
    cache_path = cache_path_for_cut(path, args.data_root, args.label_cache_dir)
    if cache_path.exists():
        pseudo_labels, score, metadata = load_label_cache(
            cache_path,
            expected_config=config,
            strict_config=not args.allow_label_cache_config_mismatch,
        )
    else:
        pseudo_labels, score, metadata = generate_three_class_labels(data, config)

    prediction = None
    if args.checkpoint:
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
        prediction = load_prediction(Path(args.checkpoint), data, args.crop_length, device, args.backbone)

    step = max(1, int(np.ceil(len(data) / args.max_plot_points)))
    plot_data = data[::step]
    plot_labels = pseudo_labels[::step]
    plot_score = score[::step]
    x = np.arange(0, len(data), step)[: len(plot_data)]

    normalized = (plot_data - plot_data.mean(axis=0)) / (plot_data.std(axis=0) + 1e-8)
    offsets = np.arange(normalized.shape[1]) * 5.0
    n_rows = 3 if prediction is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 8.5 if prediction is not None else 7), dpi=170, sharex=True)

    add_label_spans(axes[0], plot_labels, x)
    for i, name in enumerate(CHANNEL_NAMES):
        axes[0].plot(x, normalized[:, i] + offsets[i], linewidth=0.35)
    axes[0].set_yticks(offsets)
    axes[0].set_yticklabels(CHANNEL_NAMES)
    axes[0].set_title(f"Pseudo labels on full waveform: {path.parent.name}/{path.name}")

    axes[1].plot(x, plot_score, color="#2f6fed", linewidth=0.8)
    add_label_spans(axes[1], plot_labels, x, alpha=0.18)
    axes[1].set_ylabel("activity")
    axes[1].set_title("Activity score and rule-based pseudo labels")

    if prediction is not None:
        plot_pred = prediction[::step]
        add_label_spans(axes[2], plot_pred, x)
        axes[2].plot(x, plot_score, color="#444444", linewidth=0.6)
        axes[2].set_ylabel("prediction")
        axes[2].set_title("Model predicted process states")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[idx], alpha=0.35, label=f"{idx}: {CLASS_NAMES[idx]}")
        for idx in sorted(CLASS_NAMES)
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", frameon=False)
    axes[-1].set_xlabel("sample index in complete cut")
    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "cut_file": str(path),
        "output": str(output),
        "metadata": metadata,
        "classes": CLASS_NAMES,
        "label_cache": str(cache_path) if cache_path.exists() else None,
    }
    (output.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
