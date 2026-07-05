from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


CHANNEL_NAMES = ("Fx", "Fy", "Fz", "Vx", "Vy", "Vz", "AE")
CLASS_NAMES = {
    0: "non_cutting",
    1: "transition",
    2: "stable_cutting",
}
CLASS_COLORS = {
    0: "#d9d9d9",
    1: "#ffd166",
    2: "#8bd17c",
}


@dataclass
class PseudoLabelConfig:
    smooth_window: int = 2048
    active_threshold: float = 0.25
    transition_ratio: float = 0.05
    min_transition_points: int = 4096
    min_active_points: int = 8192


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 1 or values.size <= 2:
        return values
    use_window = int(min(max(window, 1), values.size))
    kernel = np.ones(use_window, dtype=np.float64) / use_window
    return np.convolve(values, kernel, mode="same")


def robust_minmax(values: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(values, [low_q, high_q])
    scale = max(float(high - low), 1e-12)
    return np.clip((values - low) / scale, 0.0, 1.0)


def compute_activity_score(data: np.ndarray, smooth_window: int = 2048) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    force_mag = np.sqrt(np.sum(np.square(data[:, 0:3]), axis=1))
    vib_mag = np.sqrt(np.sum(np.square(data[:, 3:6]), axis=1))
    ae_mag = np.abs(data[:, 6])

    force_energy = np.sqrt(moving_average(np.square(force_mag), smooth_window))
    vib_energy = np.sqrt(moving_average(np.square(vib_mag), smooth_window))
    ae_energy = moving_average(ae_mag, smooth_window)

    score = (
        robust_minmax(force_energy)
        + robust_minmax(vib_energy)
        + robust_minmax(ae_energy)
    ) / 3.0
    return moving_average(score, max(16, smooth_window // 4))


def _longest_true_region(mask: np.ndarray) -> Tuple[int, int]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0, len(mask)

    padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant")
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(starts[best]), int(ends[best])


def generate_three_class_labels(
    data: np.ndarray,
    config: PseudoLabelConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    config = config or PseudoLabelConfig()
    score = compute_activity_score(data, smooth_window=config.smooth_window)
    active_mask = score >= config.active_threshold
    active_start, active_end = _longest_true_region(active_mask)

    if active_end - active_start < config.min_active_points:
        active_start, active_end = 0, len(score)

    labels = np.zeros(len(score), dtype=np.int64)
    active_len = active_end - active_start
    transition_len = max(
        config.min_transition_points,
        int(round(active_len * config.transition_ratio)),
    )
    transition_len = min(transition_len, max(1, active_len // 2))

    entry_end = active_start + transition_len
    exit_start = active_end - transition_len
    labels[active_start:entry_end] = 1
    labels[entry_end:exit_start] = 2
    labels[exit_start:active_end] = 1

    metadata = {
        "active_start": int(active_start),
        "active_end": int(active_end),
        "transition_len": int(transition_len),
        "n_points": int(len(score)),
    }
    return labels, score.astype(np.float32), metadata


def class_counts(labels: np.ndarray) -> Dict[str, int]:
    labels = np.asarray(labels)
    return {
        CLASS_NAMES[int(cls)]: int((labels == cls).sum())
        for cls in sorted(CLASS_NAMES)
    }
