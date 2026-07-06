import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phm2010_segmentation.dataset import TOOLS, list_cut_files, read_cut_csv
from phm2010_segmentation.pseudo_label import (
    CHANNEL_NAMES,
    CLASS_COLORS,
    CLASS_NAMES,
    PseudoLabelConfig,
    class_counts,
    generate_three_class_labels,
)


CURRENT_DIR = Path(__file__).resolve().parent


def parse_tools(value: str) -> list[str]:
    if value.lower() == "all":
        return list(TOOLS)
    tools = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [tool for tool in tools if tool not in TOOLS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown tools: {unknown}; expected {list(TOOLS)} or all")
    return tools


def add_label_spans(ax, labels: np.ndarray, x: np.ndarray, alpha: float = 0.22):
    labels = np.asarray(labels)
    change_points = np.where(np.diff(labels) != 0)[0] + 1
    starts = np.concatenate([[0], change_points])
    ends = np.concatenate([change_points, [len(labels)]])
    for start, end in zip(starts, ends):
        label = int(labels[start])
        ax.axvspan(x[start], x[end - 1], color=CLASS_COLORS[label], alpha=alpha, linewidth=0)


def label_counts(labels: np.ndarray) -> dict[str, int]:
    counts = class_counts(labels)
    return {name: int(counts.get(name, 0)) for name in CLASS_NAMES.values()}


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


def plot_one_cut(cut_file: Path, data_root: Path, output_dir: Path, config: PseudoLabelConfig, max_plot_points: int):
    data = read_cut_csv(cut_file)
    labels, score, metadata = generate_three_class_labels(data, config)
    counts = label_counts(labels)
    n_points = int(len(labels))

    sample_id = f"{cut_file.parent.name}_{cut_file.stem}"
    output = output_dir / f"{sample_id}_labels.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    step = max(1, int(np.ceil(len(data) / max_plot_points)))
    plot_data = data[::step]
    plot_labels = labels[::step]
    plot_score = score[::step]
    x = np.arange(0, len(data), step)[: len(plot_data)]

    normalized = (plot_data - plot_data.mean(axis=0)) / (plot_data.std(axis=0) + 1e-8)
    offsets = np.arange(normalized.shape[1]) * 5.0
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), dpi=170, sharex=True)

    add_label_spans(axes[0], plot_labels, x)
    for channel_index, name in enumerate(CHANNEL_NAMES):
        axes[0].plot(x, normalized[:, channel_index] + offsets[channel_index], linewidth=0.35)
    axes[0].set_yticks(offsets)
    axes[0].set_yticklabels(CHANNEL_NAMES)
    axes[0].set_title(f"Pseudo labels on full waveform: {cut_file.parent.name}/{cut_file.name}")

    axes[1].plot(x, plot_score, color="#2f6fed", linewidth=0.8)
    add_label_spans(axes[1], plot_labels, x, alpha=0.18)
    axes[1].set_ylabel("activity")
    axes[1].set_title("Activity score and rule-based pseudo labels")

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
        "cut_file": str(cut_file),
        "relative_cut_file": str(cut_file.resolve().relative_to(data_root.resolve())),
        "output": str(output),
        "metadata": metadata,
        "classes": CLASS_NAMES,
        "class_counts": counts,
        "class_percent": {
            name: round(100.0 * count / max(n_points, 1), 6)
            for name, count in counts.items()
        },
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def make_contact_sheet(image_files: list[Path], output: Path, thumb_size=(520, 230), cols=4):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    if not image_files:
        return None

    thumb_w, thumb_h = thumb_size
    rows = (len(image_files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image_file in enumerate(image_files):
        image = Image.open(image_file).convert("RGB")
        image.thumbnail((thumb_w, thumb_h - 22), Image.LANCZOS)
        x = (index % cols) * thumb_w
        y = (index // cols) * thumb_h
        sheet.paste(image, (x + (thumb_w - image.width) // 2, y + 20))
        draw.text((x + 8, y + 4), image_file.stem.replace("_labels", ""), fill=(0, 0, 0))

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)
    return output


def write_summary_csv(rows: list[dict], output: Path):
    fieldnames = [
        "sample",
        "cut_file",
        "n_points",
        "active_start",
        "active_end",
        "non_cutting",
        "transition",
        "stable_cutting",
        "non_cutting_percent",
        "transition_percent",
        "stable_cutting_percent",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Randomly visualize PHM2010 pseudo-label samples.")
    parser.add_argument("--data-root", default=str(ROOT / "PHM 2010"))
    parser.add_argument("--output-root", default=str(CURRENT_DIR / "outputs" / "random_label_samples"))
    parser.add_argument("--tools", type=parse_tools, default=list(TOOLS))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-plot-points", type=int, default=25000)
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
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    rng = random.Random(args.seed)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.seed is not None and args.run_name is None:
        run_name = f"{run_name}_seed{args.seed}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_cut_files(data_root, tools=args.tools)
    if not files:
        raise ValueError(f"No cut files found under {data_root} for tools={args.tools}")
    sample_count = min(args.samples, len(files))
    selected_files = rng.sample(files, sample_count)
    selected_files = sorted(selected_files, key=lambda path: (path.parent.name, path.name))

    config = make_config(args)
    summaries = [
        plot_one_cut(cut_file, data_root, output_dir, config, args.max_plot_points)
        for cut_file in selected_files
    ]

    rows = []
    for summary in summaries:
        metadata = summary["metadata"]
        counts = summary["class_counts"]
        percents = summary["class_percent"]
        rows.append(
            {
                "sample": Path(summary["output"]).stem.replace("_labels", ""),
                "cut_file": summary["relative_cut_file"],
                "n_points": metadata["n_points"],
                "active_start": metadata["active_start"],
                "active_end": metadata["active_end"],
                "non_cutting": counts["non_cutting"],
                "transition": counts["transition"],
                "stable_cutting": counts["stable_cutting"],
                "non_cutting_percent": percents["non_cutting"],
                "transition_percent": percents["transition"],
                "stable_cutting_percent": percents["stable_cutting"],
            }
        )

    summary_csv = output_dir / "random_label_samples_summary.csv"
    write_summary_csv(rows, summary_csv)
    contact_sheet = make_contact_sheet(
        sorted(output_dir.glob("*_labels.png")),
        output_dir / "random_label_samples_contact_sheet.jpg",
    )

    run_summary = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "tools": args.tools,
        "samples": sample_count,
        "seed": args.seed,
        "summary_csv": str(summary_csv),
        "contact_sheet": str(contact_sheet) if contact_sheet is not None else None,
        "selected_files": [str(path) for path in selected_files],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
