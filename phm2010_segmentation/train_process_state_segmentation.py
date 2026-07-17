import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phm2010_segmentation.dataset import (
    PHM2010SegmentationDataset,
    TOOLS,
    load_excluded_cut_paths,
    make_tool_split,
)
from phm2010_segmentation.metrics import (
    average_sample_metrics,
    confusion_matrix_1d,
    segmentation_metrics_from_confusion,
)
from phm2010_segmentation.pseudo_label import CLASS_NAMES, PseudoLabelConfig
from src.segmentation_factory import SEGMENTATION_MODEL_NAMES, build_segmentation_model


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_EXCLUDE_SAMPLES_CSV = CURRENT_DIR / "config" / "non_cutting_exclude_samples.csv"
BINARY_CLASS_NAMES = {
    0: "transition",
    1: "stable_cutting",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def criterion(outputs, target, weight=None):
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    losses = {}
    for name, logits in outputs.items():
        loss = loss_fn(logits, target)
        losses[name] = loss if name == "out" else 0.4 * loss
    return sum(losses.values())


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_class_names(task: str) -> dict[int, str]:
    return BINARY_CLASS_NAMES if task == "binary" else CLASS_NAMES


def print_training_config(args, folds, num_classes: int) -> None:
    config = {
        "folds": list(folds),
        "num_classes": num_classes,
        "class_names": get_class_names(args.task),
        "device": "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "parameters": vars(args),
    }
    print("\n========== Training configuration ==========")
    print(json.dumps(to_jsonable(config), indent=2, ensure_ascii=False))
    print("============================================\n", flush=True)


def count_trainable_parameters(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def evaluate(model, loader, device, num_classes=3, class_names=None):
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    sample_confusions = []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            outputs = model(signals)["out"]
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
            confusion += confusion_matrix_1d(labels_np, preds, num_classes=num_classes)
            for sample_labels, sample_preds in zip(labels_np, preds):
                sample_confusions.append(
                    confusion_matrix_1d(sample_labels, sample_preds, num_classes=num_classes)
                )
    metrics = segmentation_metrics_from_confusion(confusion, class_names=class_names)
    metrics.update(average_sample_metrics(sample_confusions, class_names=class_names))
    return metrics, confusion


def make_loader(args, tools, train, excluded_cut_paths):
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
    dataset = PHM2010SegmentationDataset(
        data_root=args.data_root,
        tools=tools,
        crop_length=args.crop_length,
        train=train,
        max_cuts_per_tool=args.max_cuts_per_tool,
        pseudo_label_config=config,
        label_cache_dir=args.label_cache_dir,
        require_label_cache=not args.allow_missing_label_cache,
        strict_label_cache_config=not args.allow_label_cache_config_mismatch,
        task=args.task,
        excluded_cut_paths=excluded_cut_paths,
        eval_mode="center" if train else args.eval_mode,
        train_sampling=args.train_sampling,
        train_windows_per_cut=args.train_windows_per_cut,
        eval_windows_per_cut=args.eval_windows_per_cut,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
    )


def run_fold(args, test_tool: str):
    train_tools, val_tools, test_tools = make_tool_split(test_tool)
    output_dir = Path(args.output_dir) / f"fold_{test_tool}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    excluded_cut_paths = load_excluded_cut_paths(args.exclude_samples_csv)
    class_names = get_class_names(args.task)
    num_classes = len(class_names)

    train_loader = make_loader(args, train_tools, train=True, excluded_cut_paths=excluded_cut_paths)
    val_loader = make_loader(args, val_tools, train=False, excluded_cut_paths=excluded_cut_paths)
    test_loader = make_loader(args, test_tools, train=False, excluded_cut_paths=excluded_cut_paths)

    model = build_segmentation_model(
        name=args.model,
        in_channels=7,
        num_classes=num_classes,
        aux_loss=True,
        backbone_name=args.backbone,
    ).to(device)
    total_params, trainable_params = count_trainable_parameters(model)
    print(
        f"fold={test_tool} model={args.model} backbone={args.backbone} "
        f"total_params={total_params:,} trainable_params={trainable_params:,}",
        flush=True,
    )
    class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_state = None
    best_val_miou = -1.0
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(signals), labels, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_metrics, _ = evaluate(model, val_loader, device, num_classes=num_classes, class_names=class_names)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "val_point_accuracy": val_metrics["point_accuracy"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_sample_mean_iou": val_metrics["sample_mean_iou"],
            "val_mean_iou_all_classes": val_metrics["mean_iou_all_classes"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        rows.append(row)
        print(
            f"fold={test_tool} epoch={epoch}/{args.epochs} "
            f"loss={row['train_loss']:.4f} val_sample_mIoU={row['val_sample_mean_iou']:.4f} "
            f"val_mIoU={row['val_mean_iou']:.4f} "
            f"val_acc={row['val_point_accuracy']:.4f}"
        )
        if val_metrics["sample_mean_iou"] > best_val_miou:
            best_val_miou = val_metrics["sample_mean_iou"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics, val_confusion = evaluate(model, val_loader, device, num_classes=num_classes, class_names=class_names)
    test_metrics, test_confusion = evaluate(model, test_loader, device, num_classes=num_classes, class_names=class_names)

    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    checkpoint_path = None
    if args.save_checkpoint and best_state is not None:
        checkpoint_path = output_dir / "best_model.pth"
        torch.save(best_state, checkpoint_path)

    summary = {
        "fold": test_tool,
        "train_tools": train_tools,
        "val_tools": val_tools,
        "test_tools": test_tools,
        "task": args.task,
        "classes": class_names,
        "eval_mode": args.eval_mode,
        "excluded_samples_csv": args.exclude_samples_csv,
        "excluded_cut_count": len(excluded_cut_paths),
        "checkpoint_saved": checkpoint_path is not None,
        "checkpoint_path": checkpoint_path,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_confusion": val_confusion.tolist(),
        "test_confusion": test_confusion.tolist(),
    }
    summary = to_jsonable(summary)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="PHM 2010 process-state segmentation training.")
    parser.add_argument("--data-root", default=str(ROOT / "PHM 2010"))
    parser.add_argument("--output-dir", default=str(CURRENT_DIR / "outputs"))
    parser.add_argument("--label-cache-dir", default=str(CURRENT_DIR / "label_cache"))
    parser.add_argument("--fold", default="all", choices=list(TOOLS) + ["all"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-length", type=int, default=8192)
    parser.add_argument("--max-cuts-per-tool", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--task", default="binary", choices=["three_class", "binary"])
    parser.add_argument("--exclude-samples-csv", action="append", default=[str(DEFAULT_EXCLUDE_SAMPLES_CSV)])
    parser.add_argument("--eval-mode", default="multi_position", choices=["center", "boundary", "multi_position"])
    parser.add_argument(
        "--train-sampling",
        default="multi_position_random",
        choices=["random", "multi_position", "multi_position_random"],
    )
    parser.add_argument("--train-windows-per-cut", type=int, default=21)
    parser.add_argument("--eval-windows-per-cut", type=int, default=21)
    parser.add_argument("--model", default="deeplabv3_1d", choices=list(SEGMENTATION_MODEL_NAMES))
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "lstm"])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weights", type=float, nargs="+", default=None)
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
    parser.add_argument("--allow-missing-label-cache", action="store_true")
    parser.add_argument("--allow-label-cache-config-mismatch", action="store_true")
    parser.add_argument("--save-checkpoint", dest="save_checkpoint", action="store_true", default=True)
    parser.add_argument("--no-save-checkpoint", dest="save_checkpoint", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    num_classes = len(get_class_names(args.task))
    if args.class_weights is None:
        args.class_weights = [1.0, 1.0] if args.task == "binary" else [1.0, 1.0, 1.0]
    if len(args.class_weights) != num_classes:
        raise ValueError(
            f"--class-weights must contain {num_classes} values for task={args.task}, "
            f"got {len(args.class_weights)}"
        )
    folds = TOOLS if args.fold == "all" else (args.fold,)
    print_training_config(args, folds, num_classes=num_classes)
    summaries = [run_fold(args, fold) for fold in folds]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cross_validation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(summaries), file, indent=2)
    print(json.dumps(to_jsonable(summaries), indent=2))


if __name__ == "__main__":
    main()
