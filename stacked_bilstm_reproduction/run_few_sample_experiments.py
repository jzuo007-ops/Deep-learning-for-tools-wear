import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run few-sample Stacked-BiLSTM experiments.")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--split-mode", choices=["chronological", "case_holdout", "random", "random_run"], default="random_run")
    parser.add_argument("--sample-mode", choices=["run_sequence", "segment_sequence"], default="segment_sequence")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--n-segments", type=int, default=16)
    parser.add_argument("--segment-window", type=int, default=8)
    parser.add_argument("--segment-step", type=int, default=4)
    parser.add_argument("--no-impute-vb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="3. Milling")
    parser.add_argument("--mat-file", default="mill.mat")
    parser.add_argument("--output-dir", default=str(CURRENT_DIR / "outputs" / "few_sample_sweep"))
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    for ratio in args.ratios:
        run_dir = root / f"train_ratio_{ratio:.2f}".replace(".", "p")
        command = [
            sys.executable,
            str(CURRENT_DIR / "train_stacked_bilstm.py"),
            "--data-root",
            args.data_root,
            "--mat-file",
            args.mat_file,
            "--output-dir",
            str(run_dir),
            "--sample-mode",
            args.sample_mode,
            "--train-ratio",
            str(ratio),
            "--split-mode",
            args.split_mode,
            "--epochs",
            str(args.epochs),
            "--lookback",
            str(args.lookback),
            "--n-segments",
            str(args.n_segments),
            "--segment-window",
            str(args.segment_window),
            "--segment-step",
            str(args.segment_step),
            "--seed",
            str(args.seed),
        ]
        if args.no_impute_vb:
            command.append("--no-impute-vb")
        print("\nRunning:", " ".join(command))
        subprocess.run(command, check=True)

        with (run_dir / "summary.json").open("r", encoding="utf-8") as file:
            summary = json.load(file)
        row = {
            "train_ratio": ratio,
            "sample_mode": args.sample_mode,
            "split_mode": args.split_mode,
            "test_MAE": summary["test_metrics"]["MAE"],
            "test_RMSE": summary["test_metrics"]["RMSE"],
            "test_R2": summary["test_metrics"]["R2"],
            "test_MAPE_percent": summary["test_metrics"]["MAPE_percent"],
            "train_size": summary["split_sizes"]["train"],
            "val_size": summary["split_sizes"]["val"],
            "test_size": summary["split_sizes"]["test"],
        }
        rows.append(row)

    with (root / "few_sample_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFew-sample summary saved to: {root / 'few_sample_summary.csv'}")


if __name__ == "__main__":
    main()
