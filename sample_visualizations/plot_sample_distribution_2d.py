from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transforms as T
from my_dataset import ToolWear1DDataset
from train import stratified_split


OUTPUT_DIR = ROOT / "sample_visualizations"
OUTPUT_PNG = OUTPUT_DIR / "sample_distribution_2d.png"
OUTPUT_CSV = OUTPUT_DIR / "sample_distribution_2d.csv"


def get_center_crop(seq_length=4096):
    return T.Compose1D([T.CenterCrop1D(seq_length)])


def load_window_samples():
    data_root = ROOT / "3. Milling"
    window_size = 4096
    window_stride = 2048
    full_dataset = ToolWear1DDataset(
        str(data_root),
        mat_file="mill.mat",
        transforms=None,
        impute_missing_vb=True,
        vb_interpolation_method="cubic",
    )
    train_indices, val_indices = stratified_split(full_dataset.labels, train_ratio=0.8, seed=42)
    all_indices = train_indices + val_indices

    dataset = ToolWear1DDataset(
        str(data_root),
        mat_file="mill.mat",
        transforms=get_center_crop(seq_length=window_size),
        indices=all_indices,
        label_thresholds=(full_dataset.binary_threshold,),
        label_mode="threshold",
        impute_missing_vb=True,
        vb_interpolation_method="cubic",
        use_dwt_denoise=True,
        dwt_channels=["smcAC", "smcDC", "vib_table", "vib_spindle"],
        window_size=window_size,
        window_stride=window_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    features = []
    labels = []
    with torch.no_grad():
        for signals, targets in loader:
            features.append(signals.flatten(start_dim=1).numpy())
            labels.append(targets.numpy())

    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        full_dataset.binary_threshold,
        len(train_indices),
        len(val_indices),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, labels, threshold, n_train_runs, n_val_runs = load_window_samples()

    x_scaled = StandardScaler().fit_transform(x)
    pca_50 = PCA(n_components=min(50, x_scaled.shape[0] - 1, x_scaled.shape[1]), random_state=42)
    x_pca_50 = pca_50.fit_transform(x_scaled)

    pca_2 = PCA(n_components=2, random_state=42)
    coords_pca = pca_2.fit_transform(x_scaled)
    coords_tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
        max_iter=1200,
    ).fit_transform(x_pca_50)

    colors = {0: "#2f6fed", 1: "#d64545"}
    names = {0: "lower wear", 1: "high wear"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=180)
    for ax, coords, title in [
        (axes[0], coords_pca, "PCA 2D"),
        (axes[1], coords_tsne, "t-SNE 2D after PCA-50"),
    ]:
        for cls in [0, 1]:
            mask = labels == cls
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=18,
                c=colors[cls],
                label=f"{names[cls]} (n={int(mask.sum())})",
                alpha=0.75,
                edgecolors="none",
            )
        ax.set_title(title)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)
        ax.legend(frameon=False, fontsize=8)

    explained = float(np.sum(pca_2.explained_variance_ratio_) * 100.0)
    fig.suptitle(
        "Tool-wear window sample distribution "
        f"(cubic VB + DWT, threshold={threshold:.3f}, PCA-2 var={explained:.2f}%)",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        f"Samples: {len(labels)} windows from {n_train_runs + n_val_runs} runs; "
        f"window=4096, stride=2048; classes from VB threshold.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close(fig)

    rows = np.column_stack([labels, coords_pca, coords_tsne])
    header = "label,pca_x,pca_y,tsne_x,tsne_y"
    np.savetxt(OUTPUT_CSV, rows, delimiter=",", header=header, comments="", fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"])

    print(f"Saved figure: {OUTPUT_PNG}")
    print(f"Saved coordinates: {OUTPUT_CSV}")
    print(f"Samples: {len(labels)}")
    print(f"Class counts: lower={int((labels == 0).sum())}, high={int((labels == 1).sum())}")


if __name__ == "__main__":
    main()
