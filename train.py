import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import transforms as T
from my_dataset import ToolWear1DDataset
from src.deeplabv3_model import DeepLabV3_1D


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform(train=True, seq_length=4096, channel_means=None, channel_stds=None):
    if channel_means is None:
        channel_means = [0.0] * 6
    if channel_stds is None:
        channel_stds = [1.0] * 6

    crop = T.RandomCrop1D(seq_length) if train else T.CenterCrop1D(seq_length)
    return T.Compose1D([
        crop,
        T.Normalize1D(mean=channel_means, std=channel_stds),
    ])


def get_stats_transform(seq_length=4096):
    return T.Compose1D([T.CenterCrop1D(seq_length)])


def compute_channel_stats(dataset, batch_size=64):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    total_sum = None
    total_square_sum = None
    total_count = 0

    with torch.no_grad():
        for signals, _ in loader:
            signals = signals.float()
            batch_sum = signals.sum(dim=(0, 2))
            batch_square_sum = signals.square().sum(dim=(0, 2))
            batch_count = signals.shape[0] * signals.shape[2]

            if total_sum is None:
                total_sum = batch_sum
                total_square_sum = batch_square_sum
            else:
                total_sum += batch_sum
                total_square_sum += batch_square_sum

            total_count += batch_count

    channel_means = total_sum / total_count
    channel_vars = total_square_sum / total_count - channel_means.square()
    channel_stds = torch.sqrt(torch.clamp(channel_vars, min=1e-6))
    channel_stds = torch.clamp(channel_stds, min=1e-3)
    return channel_means.cpu().tolist(), channel_stds.cpu().tolist()


def criterion(inputs, target, weight=None):
    losses = {}
    loss_func = nn.CrossEntropyLoss(weight=weight)
    for name, logits in inputs.items():
        loss = loss_func(logits, target)
        losses[name] = loss if name == "out" else loss * 0.4
    return sum(losses.values())


def evaluate(model, loader, device):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for signals, targets in loader:
            signals = signals.to(device)
            targets = targets.to(device)
            outputs = model(signals)
            logits = outputs["out"]
            preds = logits.argmax(dim=1)
            predictions.append(preds.cpu())
            labels.append(targets.cpu())

    predictions = torch.cat(predictions).numpy()
    labels = torch.cat(labels).numpy()

    accuracy = (predictions == labels).mean()
    classes = np.unique(np.concatenate([labels, predictions]))
    f1_scores = []
    for cls in classes:
        tp = ((predictions == cls) & (labels == cls)).sum()
        fp = ((predictions == cls) & (labels != cls)).sum()
        fn = ((predictions != cls) & (labels == cls)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for y_true, y_pred in zip(labels, predictions):
        confusion[class_to_index[y_true], class_to_index[y_pred]] += 1

    return float(accuracy), float(np.mean(f1_scores)), confusion, classes


def stratified_split(labels, train_ratio=0.8, seed=42):
    labels = np.asarray(labels)
    train_indices = []
    val_indices = []
    rng = np.random.default_rng(seed)

    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        split_idx = int(len(cls_indices) * train_ratio)
        train_indices.extend(cls_indices[:split_idx].tolist())
        val_indices.extend(cls_indices[split_idx:].tolist())

    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices.tolist(), val_indices.tolist()


def main():
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = os.path.join(".", "3. Milling")
    batch_size = 16
    use_vb_interpolation = True
    vb_interpolation_method = "cubic"
    use_dwt_denoise = True
    dwt_channels = ["smcAC", "smcDC", "vib_table", "vib_spindle"]
    window_size = 4096
    window_stride = 2048
    seq_length = window_size
    epochs = 30
    num_classes = 2

    full_dataset = ToolWear1DDataset(
        data_root,
        mat_file="mill.mat",
        transforms=None,
        impute_missing_vb=use_vb_interpolation,
        vb_interpolation_method=vb_interpolation_method,
    )
    train_indices, val_indices = stratified_split(full_dataset.labels, train_ratio=0.8, seed=42)
    label_thresholds = (full_dataset.binary_threshold,)

    print(f"Binary VB threshold: {label_thresholds[0]:.6f}")
    print(f"VB interpolation: {use_vb_interpolation}")
    print(f"VB interpolation method: {vb_interpolation_method}")
    print(f"DWT denoise: {use_dwt_denoise}")
    print(f"DWT channels: {dwt_channels}")
    print(f"Window size/stride: {window_size}/{window_stride}")
    print(f"Train/val runs: {len(train_indices)}/{len(val_indices)}")

    stats_dataset = ToolWear1DDataset(
        data_root,
        mat_file="mill.mat",
        transforms=get_stats_transform(seq_length=seq_length),
        indices=train_indices,
        label_thresholds=label_thresholds,
        label_mode="threshold",
        impute_missing_vb=use_vb_interpolation,
        vb_interpolation_method=vb_interpolation_method,
        use_dwt_denoise=use_dwt_denoise,
        dwt_channels=dwt_channels,
        window_size=window_size,
        window_stride=window_stride,
    )
    channel_means, channel_stds = compute_channel_stats(stats_dataset, batch_size=64)

    train_dataset = ToolWear1DDataset(
        data_root,
        mat_file="mill.mat",
        transforms=get_transform(
            train=True,
            seq_length=seq_length,
            channel_means=channel_means,
            channel_stds=channel_stds,
        ),
        indices=train_indices,
        label_thresholds=label_thresholds,
        label_mode="threshold",
        impute_missing_vb=use_vb_interpolation,
        vb_interpolation_method=vb_interpolation_method,
        use_dwt_denoise=use_dwt_denoise,
        dwt_channels=dwt_channels,
        window_size=window_size,
        window_stride=window_stride,
    )
    val_dataset = ToolWear1DDataset(
        data_root,
        mat_file="mill.mat",
        transforms=get_transform(
            train=False,
            seq_length=seq_length,
            channel_means=channel_means,
            channel_stds=channel_stds,
        ),
        indices=val_indices,
        label_thresholds=label_thresholds,
        label_mode="threshold",
        impute_missing_vb=use_vb_interpolation,
        vb_interpolation_method=vb_interpolation_method,
        use_dwt_denoise=use_dwt_denoise,
        dwt_channels=dwt_channels,
        window_size=window_size,
        window_stride=window_stride,
    )
    print(f"Train/val window samples: {len(train_dataset)}/{len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=val_dataset.collate_fn,
    )

    model = DeepLabV3_1D(in_channels=6, num_classes=num_classes, aux_loss=True, classification=True)
    model.to(device)

    train_labels = np.asarray(train_dataset.labels)
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    class_weights = torch.tensor(
        class_counts.max() / np.maximum(class_counts, 1),
        dtype=torch.float32,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state_dict = None
    best_val_f1 = -1.0
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        val_acc, val_f1, confusion, classes = evaluate(model, val_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"loss={avg_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    final_model = DeepLabV3_1D(in_channels=6, num_classes=num_classes, aux_loss=True, classification=True)
    final_model.load_state_dict(best_state_dict)
    final_model.to(device)
    final_model.eval()

    test_acc, test_f1, confusion, classes = evaluate(final_model, val_loader, device)
    print("\nEvaluation on validation set")
    print("Accuracy:", round(test_acc, 4))
    print("Macro F1:", round(test_f1, 4))
    print("Best validation Accuracy:", round(best_val_acc, 4))
    print("Best validation Macro F1:", round(best_val_f1, 4))
    print("Classes:", classes)
    print("Confusion matrix:")
    print(confusion)

    with open("training_results.txt", "a", encoding="utf-8") as file:
        file.write("\n1D DeepLabV3-ResNet50 classification: cubic VB interpolation + DWT\n")
        file.write(f"Device: {device}\n")
        file.write(f"VB interpolation: {use_vb_interpolation}\n")
        file.write(f"VB interpolation method: {vb_interpolation_method}\n")
        file.write(f"DWT denoise: {use_dwt_denoise}\n")
        file.write(f"DWT channels: {', '.join(dwt_channels)}\n")
        file.write(f"Window size/stride: {window_size}/{window_stride}\n")
        file.write(f"Binary VB threshold: {label_thresholds[0]:.6f}\n")
        file.write(f"Train/val runs: {len(train_indices)}/{len(val_indices)}\n")
        file.write(f"Train/val window samples: {len(train_dataset)}/{len(val_dataset)}\n")
        file.write(f"Accuracy: {test_acc:.4f}\n")
        file.write(f"Macro F1: {test_f1:.4f}\n")
        file.write(f"Best validation Accuracy: {best_val_acc:.4f}\n")
        file.write(f"Best validation Macro F1: {best_val_f1:.4f}\n")
        file.write(f"Classes: {classes.tolist()}\n")
        file.write("Confusion matrix:\n")
        file.write(np.array2string(confusion))
        file.write("\n")


if __name__ == "__main__":
    main()
