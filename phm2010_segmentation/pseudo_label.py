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
    inactive_threshold: float = 0.12
    transition_ratio: float = 0.05
    min_transition_points: int = 4096
    min_active_points: int = 8192
    min_cut_ratio: float = 0.35
    max_gap_ratio: float = 0.03
    max_gap_points: int = 8192
    edge_margin_ratio: float = 0.01


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


def _fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or mask.size == 0:
        return mask

    padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant", constant_values=1)
    changes = np.diff(padded)
    starts = np.where(changes == -1)[0]
    ends = np.where(changes == 1)[0]
    for start, end in zip(starts, ends):
        if end - start <= max_gap:
            mask[start:end] = True
    return mask


def _remove_short_edge_regions(mask: np.ndarray, edge_margin: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if edge_margin <= 0 or mask.size == 0:
        return mask

    padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant")
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for start, end in zip(starts, ends):
        touches_left = start <= edge_margin
        touches_right = end >= len(mask) - edge_margin
        if (touches_left or touches_right) and end - start <= edge_margin:
            mask[start:end] = False
    return mask


def _true_regions(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant")
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return starts, ends


def _region_overlapping(starts: np.ndarray, ends: np.ndarray, start: int, end: int) -> Tuple[int, int] | None:
    best_region = None
    best_overlap = 0
    for region_start, region_end in zip(starts, ends):
        overlap = max(0, min(int(region_end), end) - max(int(region_start), start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_region = (int(region_start), int(region_end))
    return best_region


def detect_cutting_region(score: np.ndarray, config: PseudoLabelConfig) -> Tuple[int, int, np.ndarray]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    n_points = len(score)
    if n_points == 0:
        return 0, 0, np.zeros(0, dtype=bool)

    high_mask = score >= config.active_threshold
    low_mask = score >= config.inactive_threshold
    if not high_mask.any():
        low_mask = score >= np.percentile(score, 35)

    max_gap = max(
        config.min_transition_points,
        min(config.max_gap_points, int(round(n_points * config.max_gap_ratio))),
    )
    candidate_mask = _fill_short_false_gaps(low_mask, max_gap=max_gap)
    candidate_mask = _remove_short_edge_regions(
        candidate_mask,
        edge_margin=int(round(n_points * config.edge_margin_ratio)),
    )

    if high_mask.any():
        high_start, high_end = _longest_true_region(high_mask)
        starts, ends = _true_regions(candidate_mask)
        active_region = _region_overlapping(starts, ends, high_start, high_end)
        if active_region is None:
            active_start, active_end = high_start, high_end
        else:
            active_start, active_end = active_region
    else:
        true_indices = np.where(candidate_mask)[0]
        active_start = int(true_indices[0]) if true_indices.size else 0
        active_end = int(true_indices[-1]) + 1 if true_indices.size else n_points

    active_len = active_end - active_start
    min_cut_points = max(config.min_active_points, int(round(n_points * config.min_cut_ratio)))
    if active_len < min_cut_points:
        active_start, active_end = 0, n_points

    active_mask = np.zeros(n_points, dtype=bool)
    active_mask[active_start:active_end] = True
    return active_start, active_end, active_mask


def generate_three_class_labels(
    data: np.ndarray,
    config: PseudoLabelConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    config = config or PseudoLabelConfig()
    score = compute_activity_score(data, smooth_window=config.smooth_window)
    active_start, active_end, active_mask = detect_cutting_region(score, config)

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
        "active_points": int(active_mask.sum()),
        "active_threshold": float(config.active_threshold),
        "inactive_threshold": float(config.inactive_threshold),
        "max_gap_ratio": float(config.max_gap_ratio),
        "max_gap_points": int(config.max_gap_points),
    }
    return labels, score.astype(np.float32), metadata


def class_counts(labels: np.ndarray) -> Dict[str, int]:
    labels = np.asarray(labels)
    return {
        CLASS_NAMES[int(cls)]: int((labels == cls).sum())
        for cls in sorted(CLASS_NAMES)
    }
