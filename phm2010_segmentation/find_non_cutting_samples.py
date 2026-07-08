import argparse
import csv
import json
import shutil
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
from phm2010_segmentation.label_cache import cache_path_for_cut, load_label_cache
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


def add_label_spans(ax, labels: np.ndarray, x: np.ndarray, alpha: float = 0.22) -> None:
    labels = np.asarray(labels)
    change_points = np.where(np.diff(labels) != 0)[0] + 1
    starts = np.concatenate([[0], change_points])
    ends = np.concatenate([change_points, [len(labels)]])
    for start, end in zip(starts, ends):
        label = int(labels[start])
        ax.axvspan(x[start], x[end - 1], color=CLASS_COLORS[label], alpha=alpha, linewidth=0)


def load_or_generate_labels(
    cut_file: Path,
    data_root: Path,
    label_cache_dir: Path,
    config: PseudoLabelConfig,
    strict_config: bool,
    cache_only: bool,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray | None, bool]:
    cache_path = cache_path_for_cut(cut_file, data_root, label_cache_dir)
    if cache_path.exists():
        labels, score, metadata = load_label_cache(
            cache_path,
            expected_config=config,
            strict_config=strict_config,
        )
        return labels, score, metadata, None, True

    if cache_only:
        raise FileNotFoundError(f"Missing label cache for {cut_file}: {cache_path}")

    data = read_cut_csv(cut_file)
    labels, score, metadata = generate_three_class_labels(data, config)
    return labels, score, metadata, data, False


def plot_non_cutting_sample(
    cut_file: Path,
    data_root: Path,
    output_dir: Path,
    labels: np.ndarray,
    score: np.ndarray,
    data: np.ndarray | None,
    max_plot_points: int,
) -> Path:
    if data is None:
        data = read_cut_csv(cut_file)

    sample_id = f"{cut_file.parent.name}_{cut_file.stem}"
    output = output_dir / "plots" / f"{sample_id}_non_cutting.png"
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
    axes[0].set_title(f"Non-cutting candidate: {cut_file.parent.name}/{cut_file.name}")

    axes[1].plot(x, plot_score, color="#2f6fed", linewidth=0.8)
    add_label_spans(axes[1], plot_labels, x, alpha=0.18)
    axes[1].set_ylabel("activity")
    axes[1].set_title("Activity score and pseudo labels")

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
    return output


def make_contact_sheet(image_files: list[Path], output: Path, thumb_size=(520, 230), cols=4) -> str | None:
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
        draw.text((x + 8, y + 4), image_file.stem.replace("_non_cutting", ""), fill=(0, 0, 0))

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)
    return str(output)


def write_summary_csv(rows: list[dict], output: Path) -> None:
    fieldnames = [
        "rank",
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
        "label_cache_used",
        "plot_file",
        "copied_csv",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Find PHM2010 cuts that contain non_cutting pseudo labels.")
    parser.add_argument("--data-root", default=str(ROOT / "PHM 2010"))
    parser.add_argument("--label-cache-dir", default=str(CURRENT_DIR / "label_cache"))
    parser.add_argument("--output-root", default=str(CURRENT_DIR / "outputs" / "non_cutting_samples"))
    parser.add_argument("--tools", type=parse_tools, default=list(TOOLS))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-cuts-per-tool", type=int, default=None)
    parser.add_argument("--min-non-cutting-points", type=int, default=1)
    parser.add_argument("--min-non-cutting-percent", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stop-after-matches", type=int, default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--copy-csv", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--allow-label-cache-config-mismatch", action="store_true")
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
    label_cache_dir = Path(args.label_cache_dir)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_config(args)
    cut_files = list_cut_files(
        data_root,
        tools=args.tools,
        max_cuts_per_tool=args.max_cuts_per_tool,
    )
    if not cut_files:
        raise ValueError(f"No cut files found under {data_root} for tools={args.tools}")

    candidates = []
    skipped_missing_cache = 0
    for index, cut_file in enumerate(cut_files, start=1):
        if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0):
            print(f"[{index}/{len(cut_files)}] scanning {cut_file.parent.name}/{cut_file.name}", flush=True)
        try:
            labels, score, metadata, data, cache_used = load_or_generate_labels(
                cut_file=cut_file,
                data_root=data_root,
                label_cache_dir=label_cache_dir,
                config=config,
                strict_config=not args.allow_label_cache_config_mismatch,
                cache_only=args.cache_only,
            )
        except FileNotFoundError:
            skipped_missing_cache += 1
            continue
        counts = class_counts(labels)
        n_points = int(len(labels))
        non_cutting = int(counts["non_cutting"])
        non_cutting_percent = 100.0 * non_cutting / max(n_points, 1)
        if (
            non_cutting < args.min_non_cutting_points
            or non_cutting_percent < args.min_non_cutting_percent
        ):
            continue

        candidates.append(
            {
                "cut_file": cut_file,
                "labels": labels,
                "score": score,
                "metadata": metadata,
                "data": data,
                "cache_used": cache_used,
                "counts": counts,
                "non_cutting_percent": non_cutting_percent,
            }
        )
        print(
            f"[{index}/{len(cut_files)}] found {cut_file.parent.name}/{cut_file.name}: "
            f"{non_cutting_percent:.2f}%",
            flush=True,
        )
        if args.stop_after_matches is not None and len(candidates) >= args.stop_after_matches:
            print(f"Stopping after {len(candidates)} matches.", flush=True)
            break

    candidates.sort(
        key=lambda item: (
            -float(item["non_cutting_percent"]),
            item["cut_file"].parent.name,
            item["cut_file"].name,
        )
    )
    if args.limit is not None:
        candidates = candidates[: args.limit]

    rows = []
    image_files = []
    csv_dir = output_dir / "csv"
    json_dir = output_dir / "metadata"
    json_dir.mkdir(parents=True, exist_ok=True)

    for rank, item in enumerate(candidates, start=1):
        cut_file = item["cut_file"]
        metadata = item["metadata"]
        counts = item["counts"]
        n_points = int(metadata["n_points"])
        sample_id = f"{cut_file.parent.name}_{cut_file.stem}"

        plot_file = None
        if not args.no_plots:
            plot_file = plot_non_cutting_sample(
                cut_file=cut_file,
                data_root=data_root,
                output_dir=output_dir,
                labels=item["labels"],
                score=item["score"],
                data=item["data"],
                max_plot_points=args.max_plot_points,
            )
            image_files.append(plot_file)

        copied_csv = None
        if args.copy_csv:
            csv_dir.mkdir(parents=True, exist_ok=True)
            copied_csv = csv_dir / f"{sample_id}.csv"
            shutil.copy2(cut_file, copied_csv)

        summary = {
            "rank": rank,
            "sample": sample_id,
            "cut_file": str(cut_file),
            "relative_cut_file": str(cut_file.resolve().relative_to(data_root.resolve())),
            "metadata": metadata,
            "classes": CLASS_NAMES,
            "class_counts": counts,
            "class_percent": {
                name: round(100.0 * int(count) / max(n_points, 1), 6)
                for name, count in counts.items()
            },
            "label_cache_used": bool(item["cache_used"]),
            "plot_file": str(plot_file) if plot_file is not None else None,
            "copied_csv": str(copied_csv) if copied_csv is not None else None,
        }
        (json_dir / f"{rank:03d}_{sample_id}.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        rows.append(
            {
                "rank": rank,
                "sample": sample_id,
                "cut_file": summary["relative_cut_file"],
                "n_points": n_points,
                "active_start": metadata["active_start"],
                "active_end": metadata["active_end"],
                "non_cutting": counts["non_cutting"],
                "transition": counts["transition"],
                "stable_cutting": counts["stable_cutting"],
                "non_cutting_percent": summary["class_percent"]["non_cutting"],
                "transition_percent": summary["class_percent"]["transition"],
                "stable_cutting_percent": summary["class_percent"]["stable_cutting"],
                "label_cache_used": bool(item["cache_used"]),
                "plot_file": str(plot_file) if plot_file is not None else "",
                "copied_csv": str(copied_csv) if copied_csv is not None else "",
            }
        )

    summary_csv = output_dir / "non_cutting_samples.csv"
    write_summary_csv(rows, summary_csv)
    contact_sheet = None
    if image_files:
        contact_sheet = make_contact_sheet(
            image_files,
            output_dir / "non_cutting_samples_contact_sheet.jpg",
        )

    run_summary = {
        "data_root": str(data_root),
        "label_cache_dir": str(label_cache_dir),
        "output_dir": str(output_dir),
        "tools": args.tools,
        "scanned_files": len(cut_files),
        "matched_files": len(candidates),
        "stop_after_matches": args.stop_after_matches,
        "skipped_missing_cache": skipped_missing_cache,
        "cache_only": bool(args.cache_only),
        "min_non_cutting_points": args.min_non_cutting_points,
        "min_non_cutting_percent": args.min_non_cutting_percent,
        "summary_csv": str(summary_csv),
        "contact_sheet": contact_sheet,
        "copy_csv": bool(args.copy_csv),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
