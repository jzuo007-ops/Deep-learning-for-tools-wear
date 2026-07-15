import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phm2010_segmentation.dataset import (
    TOOLS,
    list_cut_files,
    load_excluded_cut_paths,
    normalize_window,
    read_cut_csv,
)
from phm2010_segmentation.label_cache import cache_path_for_cut, load_label_cache
from phm2010_segmentation.pseudo_label import (
    CHANNEL_NAMES,
    PseudoLabelConfig,
    generate_three_class_labels,
)
from src.segmentation_factory import SEGMENTATION_MODEL_NAMES, build_segmentation_model


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_EXCLUDE_SAMPLES_CSV = CURRENT_DIR / "config" / "non_cutting_exclude_samples.csv"
BINARY_CLASS_NAMES = {
    0: "transition",
    1: "stable_cutting",
}
BINARY_CLASS_COLORS = {
    0: "#ffd166",
    1: "#8bd17c",
}


def parse_folds(value: str) -> list[str]:
    if value.lower() == "all":
        return list(TOOLS)
    folds = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [fold for fold in folds if fold not in TOOLS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown folds: {unknown}; expected {list(TOOLS)} or all")
    return folds


def make_config(args) -> PseudoLabelConfig:
    return PseudoLabelConfig(
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


def map_three_class_to_binary(labels: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(labels) == 2, 1, 0).astype(np.int64)


def load_binary_rule_labels(
    cut_file: Path,
    data: np.ndarray,
    data_root: Path,
    label_cache_dir: Path,
    config: PseudoLabelConfig,
    strict_config: bool,
) -> tuple[np.ndarray, np.ndarray, dict, str | None]:
    cache_path = cache_path_for_cut(cut_file, data_root, label_cache_dir)
    if cache_path.exists():
        try:
            labels, score, metadata = load_label_cache(
                cache_path,
                expected_config=config,
                strict_config=strict_config,
            )
            return map_three_class_to_binary(labels), score, metadata, str(cache_path)
        except ValueError:
            if strict_config:
                labels, score, metadata = generate_three_class_labels(data, config)
                metadata["label_cache_warning"] = f"ignored stale cache: {cache_path}"
                return map_three_class_to_binary(labels), score, metadata, None
            raise

    labels, score, metadata = generate_three_class_labels(data, config)
    return map_three_class_to_binary(labels), score, metadata, None


def prediction_from_checkpoint(
    checkpoint: Path,
    data: np.ndarray,
    model_name: str,
    backbone: str,
    crop_length: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    model = build_segmentation_model(
        name=model_name,
        in_channels=7,
        num_classes=2,
        aux_loss=True,
        backbone_name=backbone,
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    n_points = len(data)
    vote_scores = np.zeros((2, n_points), dtype=np.float32)
    counts = np.zeros(n_points, dtype=np.float32)
    stride = max(1, int(stride))
    starts = list(range(0, max(1, n_points - crop_length + 1), stride))
    last_start = max(0, n_points - crop_length)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    with torch.no_grad():
        for start in starts:
            end = min(start + crop_length, n_points)
            window = data[start:end]
            if len(window) < crop_length:
                window = np.pad(window, ((0, crop_length - len(window)), (0, 0)), mode="edge")
            signal = torch.from_numpy(normalize_window(window).T).float().unsqueeze(0).to(device)
            logits = model(signal)["out"][0, :, : end - start]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            vote_scores[:, start:end] += probs
            counts[start:end] += 1.0

    vote_scores /= np.maximum(counts[None, :], 1.0)
    return vote_scores.argmax(axis=0).astype(np.int64)


def add_label_spans(ax, labels: np.ndarray, x: np.ndarray, colors: dict[int, str], alpha: float = 0.22) -> None:
    labels = np.asarray(labels)
    if labels.size == 0:
        return
    change_points = np.where(np.diff(labels) != 0)[0] + 1
    starts = np.concatenate([[0], change_points])
    ends = np.concatenate([change_points, [len(labels)]])
    for start, end in zip(starts, ends):
        label = int(labels[start])
        ax.axvspan(x[start], x[end - 1], color=colors[label], alpha=alpha, linewidth=0)


def class_percent(labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    total = max(int(labels.size), 1)
    return {
        BINARY_CLASS_NAMES[index]: round(100.0 * int((labels == index).sum()) / total, 6)
        for index in sorted(BINARY_CLASS_NAMES)
    }


def plot_one_cut(
    cut_file: Path,
    data_root: Path,
    output_dir: Path,
    checkpoint: Path,
    fold: str,
    args,
    device: torch.device,
) -> dict:
    data = read_cut_csv(cut_file)
    config = make_config(args)
    rule_labels, score, metadata, label_cache = load_binary_rule_labels(
        cut_file=cut_file,
        data=data,
        data_root=data_root,
        label_cache_dir=Path(args.label_cache_dir),
        config=config,
        strict_config=not args.allow_label_cache_config_mismatch,
    )
    pred_labels = prediction_from_checkpoint(
        checkpoint=checkpoint,
        data=data,
        model_name=args.model,
        backbone=args.backbone,
        crop_length=args.crop_length,
        stride=args.stride,
        device=device,
    )

    step = max(1, int(math.ceil(len(data) / args.max_plot_points)))
    plot_data = data[::step]
    plot_rule = rule_labels[::step]
    plot_pred = pred_labels[::step]
    plot_score = score[::step]
    x = np.arange(0, len(data), step)[: len(plot_data)]

    normalized = (plot_data - plot_data.mean(axis=0)) / (plot_data.std(axis=0) + 1e-8)
    offsets = np.arange(normalized.shape[1]) * 5.0

    fig, axes = plt.subplots(3, 1, figsize=(16, 9.5), dpi=170, sharex=True)
    add_label_spans(axes[0], plot_rule, x, BINARY_CLASS_COLORS, alpha=0.2)
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        axes[0].plot(x, normalized[:, channel_index] + offsets[channel_index], linewidth=0.32)
    axes[0].set_yticks(offsets)
    axes[0].set_yticklabels(CHANNEL_NAMES)
    axes[0].set_title(f"{fold}: rule labels on full waveform - {cut_file.parent.name}/{cut_file.name}")

    axes[1].plot(x, plot_score, color="#2f6fed", linewidth=0.8)
    add_label_spans(axes[1], plot_rule, x, BINARY_CLASS_COLORS, alpha=0.18)
    axes[1].set_ylabel("activity")
    axes[1].set_title("Activity score and rule labels")

    add_label_spans(axes[2], plot_pred, x, BINARY_CLASS_COLORS, alpha=0.28)
    axes[2].plot(x, plot_score, color="#444444", linewidth=0.55)
    axes[2].set_ylabel("prediction")
    axes[2].set_title("Model predicted process states")

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=BINARY_CLASS_COLORS[index],
            alpha=0.38,
            label=f"{index}: {BINARY_CLASS_NAMES[index]}",
        )
        for index in sorted(BINARY_CLASS_NAMES)
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", frameon=False)
    axes[-1].set_xlabel("sample index in complete cut")
    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.35)

    sample_id = f"{fold}_{cut_file.parent.name}_{cut_file.stem}"
    output = output_dir / f"{sample_id}_full_prediction.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "fold": fold,
        "cut_file": str(cut_file),
        "relative_cut_file": str(cut_file.resolve().relative_to(data_root.resolve())),
        "checkpoint": str(checkpoint),
        "output": str(output),
        "label_cache": label_cache,
        "metadata": metadata,
        "classes": BINARY_CLASS_NAMES,
        "rule_percent": class_percent(rule_labels),
        "prediction_percent": class_percent(pred_labels),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evenly_spaced_files(files: list[Path], count: int) -> list[Path]:
    if count <= 0 or count >= len(files):
        return files
    indices = np.linspace(0, len(files) - 1, num=count, dtype=int)
    return [files[int(index)] for index in indices]


def make_contact_sheet(image_files: list[Path], output: Path, thumb_size=(520, 240), cols=3) -> str | None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    if not image_files:
        return None

    thumb_w, thumb_h = thumb_size
    rows = int(math.ceil(len(image_files) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image_file in enumerate(image_files):
        image = Image.open(image_file).convert("RGB")
        image.thumbnail((thumb_w, thumb_h - 22), Image.LANCZOS)
        x = (index % cols) * thumb_w
        y = (index // cols) * thumb_h
        sheet.paste(image, (x + (thumb_w - image.width) // 2, y + 20))
        draw.text((x + 8, y + 4), image_file.stem.replace("_full_prediction", ""), fill=(0, 0, 0))

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)
    return str(output)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot full-waveform PHM2010 segmentation predictions for saved folds.")
    parser.add_argument("--data-root", default=str(ROOT / "PHM 2010"))
    parser.add_argument("--output-dir", default=str(CURRENT_DIR / "outputs" / "full_waveform_predictions"))
    parser.add_argument("--fold-output-root", default=str(CURRENT_DIR / "outputs"))
    parser.add_argument("--label-cache-dir", default=str(CURRENT_DIR / "label_cache"))
    parser.add_argument("--folds", type=parse_folds, default=list(TOOLS))
    parser.add_argument("--model", default="deeplabv3_1d", choices=list(SEGMENTATION_MODEL_NAMES))
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "lstm"])
    parser.add_argument("--crop-length", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument("--samples-per-fold", type=int, default=3)
    parser.add_argument("--all-cuts", action="store_true")
    parser.add_argument("--max-plot-points", type=int, default=30000)
    parser.add_argument("--exclude-samples-csv", action="append", default=[str(DEFAULT_EXCLUDE_SAMPLES_CSV)])
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
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    fold_output_root = Path(args.fold_output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_cut_paths = load_excluded_cut_paths(args.exclude_samples_csv)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    summaries = []
    for fold in args.folds:
        checkpoint = fold_output_root / f"fold_{fold}" / "best_model.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {checkpoint}")

        files = list_cut_files(data_root, tools=[fold])
        files = [
            path for path in files
            if str(path.resolve().relative_to(data_root.resolve())).replace("\\", "/").lower()
            not in excluded_cut_paths
        ]
        selected_files = files if args.all_cuts else evenly_spaced_files(files, args.samples_per_fold)
        for cut_file in selected_files:
            print(f"plotting {fold}: {cut_file}", flush=True)
            summaries.append(
                plot_one_cut(
                    cut_file=cut_file,
                    data_root=data_root,
                    output_dir=output_dir / f"fold_{fold}",
                    checkpoint=checkpoint,
                    fold=fold,
                    args=args,
                    device=device,
                )
            )

    image_files = sorted(output_dir.glob("fold_*/*_full_prediction.png"))
    contact_sheet = make_contact_sheet(image_files, output_dir / "full_waveform_prediction_contact_sheet.jpg")
    run_summary = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "folds": args.folds,
        "model": args.model,
        "backbone": args.backbone,
        "crop_length": args.crop_length,
        "stride": args.stride,
        "samples": len(summaries),
        "contact_sheet": contact_sheet,
        "summaries": summaries,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
