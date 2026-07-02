from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from feature_extraction import MillingRun, group_runs_by_case


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    target_mean: float
    target_std: float

    def transform_features(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean) / self.target_std

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        return y * self.target_std + self.target_mean


class WearSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def build_sequence_samples(
    runs: Sequence[MillingRun],
    lookback: int = 5,
    predict_next: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = group_runs_by_case(runs)
    x_samples: List[np.ndarray] = []
    y_samples: List[float] = []
    meta_samples: List[Tuple[int, int]] = []

    for case_id, case_runs in grouped.items():
        if len(case_runs) < lookback:
            continue

        max_start = len(case_runs) - lookback
        for start in range(max_start + 1):
            end = start + lookback
            target_index = end if predict_next else end - 1
            if target_index >= len(case_runs):
                continue

            x_samples.append(np.stack([run.features for run in case_runs[start:end]], axis=0))
            y_samples.append(case_runs[target_index].vb)
            meta_samples.append((case_id, case_runs[target_index].run_id))

    if not x_samples:
        raise ValueError("No sequence samples were created. Reduce lookback or check the dataset.")

    return (
        np.stack(x_samples, axis=0).astype(np.float32),
        np.asarray(y_samples, dtype=np.float32),
        np.asarray(meta_samples, dtype=np.int64),
    )


def chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    meta: np.ndarray,
    train_ratio: float = 0.3,
    val_ratio: float = 0.2,
) -> Dict[str, np.ndarray]:
    order = np.lexsort((meta[:, 1], meta[:, 0]))
    n = len(order)
    train_end = max(1, int(n * train_ratio))
    val_end = min(n - 1, train_end + max(1, int(n * val_ratio)))

    return {
        "train": order[:train_end],
        "val": order[train_end:val_end],
        "test": order[val_end:],
    }


def case_holdout_split(
    meta: np.ndarray,
    train_ratio: float = 0.3,
    val_ratio: float = 0.2,
) -> Dict[str, np.ndarray]:
    cases = np.unique(meta[:, 0])
    cases.sort()
    n_cases = len(cases)
    train_case_end = max(1, int(n_cases * train_ratio))
    val_case_end = min(n_cases - 1, train_case_end + max(1, int(n_cases * val_ratio)))

    train_cases = set(cases[:train_case_end].tolist())
    val_cases = set(cases[train_case_end:val_case_end].tolist())
    test_cases = set(cases[val_case_end:].tolist())

    return {
        "train": np.asarray([i for i, item in enumerate(meta) if item[0] in train_cases], dtype=np.int64),
        "val": np.asarray([i for i, item in enumerate(meta) if item[0] in val_cases], dtype=np.int64),
        "test": np.asarray([i for i, item in enumerate(meta) if item[0] in test_cases], dtype=np.int64),
    }


def random_split(
    n_samples: int,
    train_ratio: float = 0.3,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.arange(n_samples)
    rng.shuffle(order)
    train_end = max(1, int(n_samples * train_ratio))
    val_end = min(n_samples - 1, train_end + max(1, int(n_samples * val_ratio)))
    return {
        "train": order[:train_end],
        "val": order[train_end:val_end],
        "test": order[val_end:],
    }


def make_split(
    x: np.ndarray,
    y: np.ndarray,
    meta: np.ndarray,
    split_mode: str = "chronological",
    train_ratio: float = 0.3,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    if split_mode == "chronological":
        return chronological_split(x, y, meta, train_ratio=train_ratio, val_ratio=val_ratio)
    if split_mode == "case_holdout":
        return case_holdout_split(meta, train_ratio=train_ratio, val_ratio=val_ratio)
    if split_mode == "random":
        return random_split(len(y), train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    raise ValueError(f"Unknown split_mode: {split_mode}")


def fit_normalizer(x_train: np.ndarray, y_train: np.ndarray) -> Normalizer:
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std = np.maximum(std, 1e-6)
    target_mean = float(y_train.mean())
    target_std = float(max(y_train.std(), 1e-6))
    return Normalizer(mean=mean, std=std, target_mean=target_mean, target_std=target_std)


def apply_normalizer(x: np.ndarray, y: np.ndarray, normalizer: Normalizer) -> Tuple[np.ndarray, np.ndarray]:
    return normalizer.transform_features(x).astype(np.float32), normalizer.transform_target(y).astype(np.float32)

