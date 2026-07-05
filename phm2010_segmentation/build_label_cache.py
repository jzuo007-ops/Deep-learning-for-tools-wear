import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phm2010_segmentation.dataset import TOOLS, list_cut_files, read_cut_csv
from phm2010_segmentation.label_cache import cache_path_for_cut, config_fingerprint, save_label_cache
from phm2010_segmentation.pseudo_label import PseudoLabelConfig, class_counts, generate_three_class_labels


CURRENT_DIR = Path(__file__).resolve().parent


def parse_tools(value: str) -> list[str]:
    if value.lower() == "all":
        return list(TOOLS)
    tools = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [tool for tool in tools if tool not in TOOLS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown tools: {unknown}; expected {list(TOOLS)} or all")
    return tools


def parse_args():
    parser = argparse.ArgumentParser(description="Build reusable PHM 2010 three-class pseudo-label cache.")
    parser.add_argument("--data-root", default=str(ROOT / "PHM 2010"))
    parser.add_argument("--label-cache-dir", default=str(CURRENT_DIR / "label_cache"))
    parser.add_argument("--tools", type=parse_tools, default=list(TOOLS))
    parser.add_argument("--max-cuts-per-tool", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=2048)
    parser.add_argument("--active-threshold", type=float, default=0.25)
    parser.add_argument("--transition-ratio", type=float, default=0.05)
    parser.add_argument("--min-transition-points", type=int, default=4096)
    parser.add_argument("--min-active-points", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    label_cache_dir = Path(args.label_cache_dir)
    config = PseudoLabelConfig(
        smooth_window=args.smooth_window,
        active_threshold=args.active_threshold,
        transition_ratio=args.transition_ratio,
        min_transition_points=args.min_transition_points,
        min_active_points=args.min_active_points,
    )
    files = list_cut_files(data_root, tools=args.tools, max_cuts_per_tool=args.max_cuts_per_tool)
    if not files:
        raise ValueError(f"No cut files found under {data_root} for tools={args.tools}")

    summary = {
        "data_root": str(data_root),
        "label_cache_dir": str(label_cache_dir),
        "config_hash": config_fingerprint(config),
        "tools": args.tools,
        "total_files": len(files),
        "built": 0,
        "skipped": 0,
        "examples": [],
    }
    for file_index, cut_file in enumerate(files, start=1):
        cache_path = cache_path_for_cut(cut_file, data_root, label_cache_dir)
        if cache_path.exists() and not args.overwrite:
            summary["skipped"] += 1
            continue

        data = read_cut_csv(cut_file)
        labels, score, metadata = generate_three_class_labels(data, config)
        save_label_cache(cache_path, labels, score, metadata, cut_file, data_root, config)
        summary["built"] += 1
        if len(summary["examples"]) < 5:
            summary["examples"].append(
                {
                    "cut_file": str(cut_file),
                    "cache_file": str(cache_path),
                    "metadata": metadata,
                    "class_counts": class_counts(labels),
                }
            )
        print(f"[{file_index}/{len(files)}] cached {cut_file} -> {cache_path}")

    label_cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = label_cache_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
