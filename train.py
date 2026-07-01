import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.deeplabv3_model import DeepLabV3_1D
from my_dataset import ToolWear1DDataset
import transforms as T


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transform(train=True, seq_length=4096, channel_means=None, channel_stds=None):
    if channel_means is None:
        channel_means = [0.0] * 6
    if channel_stds is None:
        channel_stds = [1.0] * 6
    base_transforms = [T.Normalize1D(mean=channel_means, std=channel_stds)]
    base_transforms.insert(0, T.RandomCrop1D(size=seq_length))
    return T.Compose1D(base_transforms)


def criterion(inputs, target, weight=None):
    losses = {}
    for name, x in inputs.items():
        loss_func = nn.CrossEntropyLoss(weight=weight)
        loss = loss_func(x, target)
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

    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for y_true, y_pred in zip(labels, predictions):
        confusion[int(y_true), int(y_pred)] += 1

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
    rng.shuffle(np.asarray(train_indices))
    rng.shuffle(np.asarray(val_indices))
    return train_indices, val_indices


def main():
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = os.path.join(".", "3. Milling")
    batch_size = 16
    seq_length = 8192
    epochs = 30
    num_classes = 2

    full_dataset = ToolWear1DDataset(data_root, mat_file="mill.mat", transforms=None)
    train_indices, val_indices = stratified_split(full_dataset.labels, train_ratio=0.8, seed=42)

    stats_loader = DataLoader(full_dataset, batch_size=64, shuffle=False, num_workers=0,
                              collate_fn=full_dataset.collate_fn)
    channel_means = []
    channel_stds = []
    with torch.no_grad():
        for signals, _ in stats_loader:
            signals = signals.float()
            channel_means.append(signals.mean(dim=(0, 2)))
            channel_stds.append(signals.std(dim=(0, 2)))
    channel_means = torch.stack(channel_means).mean(dim=0).cpu().tolist()
    channel_stds = torch.stack(channel_stds).mean(dim=0).cpu().tolist()
    channel_stds = [max(std, 1e-3) for std in channel_stds]

    train_dataset = ToolWear1DDataset(
        data_root,
        mat_file="mill.mat",
        transforms=get_transform(train=True, seq_length=seq_length, channel_means=channel_means, channel_stds=channel_stds),
        indices=train_indices,
    )
    val_dataset = ToolWear1DDataset(
        data_root,
        mat_file="mill.mat",
        transforms=get_transform(train=False, seq_length=seq_length, channel_means=channel_means, channel_stds=channel_stds),
        indices=val_indices,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=val_dataset.collate_fn)

    model = DeepLabV3_1D(in_channels=6, num_classes=num_classes, aux_loss=True, classification=True)
    model.to(device)

    train_labels = np.asarray(train_dataset.labels)
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    class_weights = torch.tensor(class_counts.max() / np.maximum(class_counts, 1), dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, (signals, labels) in enumerate(train_loader):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        val_acc, val_f1, confusion, classes = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch + 1}/{epochs}] loss={running_loss / len(train_loader):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_milling_classifier.pth")

    final_model = DeepLabV3_1D(in_channels=6, num_classes=num_classes, aux_loss=True, classification=True)
    final_model.load_state_dict(torch.load("best_milling_classifier.pth", map_location=device))
    final_model.to(device)
    final_model.eval()

    test_acc, test_f1, confusion, classes = evaluate(final_model, val_loader, device)
    print("\nEvaluation on validation set")
    print("Accuracy:", round(test_acc, 4))
    print("Macro F1:", round(test_f1, 4))
    print("Classes:", classes)
    print("Confusion matrix:")
    print(confusion)


if __name__ == '__main__':
    main()