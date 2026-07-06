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
    make_tool_split,
)
from phm2010_segmentation.metrics import confusion_matrix_1d, segmentation_metrics_from_confusion
from phm2010_segmentation.pseudo_label import CLASS_NAMES, PseudoLabelConfig
from src.segmentation_factory import SEGMENTATION_MODEL_NAMES, build_segmentation_model


CURRENT_DIR = Path(__file__).resolve().parent


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


def evaluate(model, loader, device, num_classes=3):
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            outputs = model(signals)["out"]
            preds = outputs.argmax(dim=1).cpu().numpy()
            confusion += confusion_matrix_1d(labels.numpy(), preds, num_classes=num_classes)
    metrics = segmentation_metrics_from_confusion(confusion)
    return metrics, confusion


def make_loader(args, tools, train):
    config = PseudoLabelConfig(
        smooth_window=args.smooth_window,
        active_threshold=args.active_threshold,
        inactive_threshold=args.inactive_threshold,
        transition_ratio=args.transition_ratio,
        min_transition_points=args.min_transition_points,
        min_active_points=args.min_active_points,
        min_cut_ratio=args.min_cut_ratio,
        max_gap_ratio=args.max_gap_ratio,
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

    train_loader = make_loader(args, train_tools, train=True)
    val_loader = make_loader(args, val_tools, train=False)
    test_loader = make_loader(args, test_tools, train=False)

    model = build_segmentation_model(
        name=args.model,
        in_channels=7,
        num_classes=3,
        aux_loss=True,
        backbone_name=args.backbone,
    ).to(device)
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

        val_metrics, _ = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "val_point_accuracy": val_metrics["point_accuracy"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        rows.append(row)
        print(
            f"fold={test_tool} epoch={epoch}/{args.epochs} "
            f"loss={row['train_loss']:.4f} val_mIoU={row['val_mean_iou']:.4f} "
            f"val_acc={row['val_point_accuracy']:.4f}"
        )
        if val_metrics["mean_iou"] > best_val_miou:
            best_val_miou = val_metrics["mean_iou"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics, val_confusion = evaluate(model, val_loader, device)
    test_metrics, test_confusion = evaluate(model, test_loader, device)

    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "fold": test_tool,
        "train_tools": train_tools,
        "val_tools": val_tools,
        "test_tools": test_tools,
        "classes": CLASS_NAMES,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_confusion": val_confusion.tolist(),
        "test_confusion": test_confusion.tolist(),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    if args.save_checkpoint and best_state is not None:
        torch.save(best_state, output_dir / "best_model.pth")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="PHM 2010 process-state segmentation training.")
    parser.add_argument("--data-root", default=str(ROOT / "PHM 2010"))
    parser.add_argument("--output-dir", default=str(CURRENT_DIR / "outputs"))
    parser.add_argument("--label-cache-dir", default=str(CURRENT_DIR / "label_cache"))
    parser.add_argument("--fold", default="c1", choices=list(TOOLS) + ["all"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-length", type=int, default=8192)
    parser.add_argument("--max-cuts-per-tool", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", default="deeplabv3_1d", choices=list(SEGMENTATION_MODEL_NAMES))
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "lstm"])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weights", type=float, nargs=3, default=[1.0, 2.0, 1.0])
    parser.add_argument("--smooth-window", type=int, default=2048)
    parser.add_argument("--active-threshold", type=float, default=0.25)
    parser.add_argument("--inactive-threshold", type=float, default=0.12)
    parser.add_argument("--transition-ratio", type=float, default=0.05)
    parser.add_argument("--min-transition-points", type=int, default=4096)
    parser.add_argument("--min-active-points", type=int, default=8192)
    parser.add_argument("--min-cut-ratio", type=float, default=0.35)
    parser.add_argument("--max-gap-ratio", type=float, default=0.20)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.01)
    parser.add_argument("--allow-missing-label-cache", action="store_true")
    parser.add_argument("--allow-label-cache-config-mismatch", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    folds = TOOLS if args.fold == "all" else (args.fold,)
    summaries = [run_fold(args, fold) for fold in folds]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cross_validation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
