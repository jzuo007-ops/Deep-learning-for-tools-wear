import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis, skew


DEFAULT_CHANNELS = (
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
)


@dataclass
class MillingRun:
    case_id: int
    run_id: int
    vb: float
    features: np.ndarray


def _scalar_field(record, name: str, default: float = np.nan) -> float:
    if name not in record.dtype.names:
        return default
    value = np.asarray(record[name]).squeeze()
    if value.size == 0:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1).astype(np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float64)

    clean = values[finite]
    clean = clean[np.abs(clean) < 1e6]
    if clean.size == 0:
        clean = values[finite]
        clean = np.sign(clean) * np.log1p(np.abs(clean))

    lower, upper = np.percentile(clean, [1.0, 99.0])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        median = np.median(clean)
        mad = np.median(np.abs(clean - median))
        scale = max(mad * 6.0, 1e-6)
        lower, upper = median - scale, median + scale

    values = np.where(np.abs(values) < 1e6, values, np.nan)
    values = np.clip(values, lower, upper)
    fill = float(np.median(clean)) if clean.size else 0.0
    values = np.nan_to_num(values, nan=fill, posinf=upper, neginf=lower)
    return values.astype(np.float64)


def _safe_div(num: float, den: float, eps: float = 1e-8) -> float:
    return float(num / (den + eps))


def time_domain_features(signal: np.ndarray) -> List[float]:
    x = _clean_signal(signal)
    abs_x = np.abs(x)
    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(np.max(abs_x))
    peak_to_peak = float(np.ptp(x))
    abs_mean = float(np.mean(abs_x))
    sqrt_abs_mean = float(np.mean(np.sqrt(abs_x + 1e-8)))
    energy = float(np.mean(np.square(x)))

    features = np.asarray([
        mean,
        std,
        rms,
        peak,
        peak_to_peak,
        abs_mean,
        float(skew(x, bias=False)) if x.size > 2 else 0.0,
        float(kurtosis(x, fisher=False, bias=False)) if x.size > 3 else 0.0,
        _safe_div(rms, abs_mean),
        _safe_div(peak, rms),
        _safe_div(peak, abs_mean),
        _safe_div(peak, sqrt_abs_mean * sqrt_abs_mean),
        energy,
    ], dtype=np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -1e6, 1e6)
    return features.astype(np.float32).tolist()


def load_milling_records(
    data_root: str,
    mat_file: str = "mill.mat",
    channels: Sequence[str] = DEFAULT_CHANNELS,
    include_process_features: bool = True,
) -> List[MillingRun]:
    mat_path = os.path.join(data_root, mat_file)
    records = loadmat(mat_path)["mill"][0]
    runs: List[MillingRun] = []

    for record in records:
        vb = _scalar_field(record, "VB")
        if not np.isfinite(vb):
            continue

        case_id = int(_scalar_field(record, "case", default=0))
        run_id = int(_scalar_field(record, "run", default=len(runs)))
        features: List[float] = []

        for channel in channels:
            if channel not in record.dtype.names:
                raise KeyError(f"Channel {channel!r} not found in mill.mat record.")
            features.extend(time_domain_features(record[channel]))

        if include_process_features:
            for name in ("DOC", "feed", "material"):
                value = _scalar_field(record, name)
                if np.isfinite(value):
                    features.append(float(np.clip(value, -1e6, 1e6)))

        runs.append(
            MillingRun(
                case_id=case_id,
                run_id=run_id,
                vb=float(vb),
                features=np.nan_to_num(
                    np.asarray(features, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
            )
        )

    runs.sort(key=lambda item: (item.case_id, item.run_id))
    return runs


def group_runs_by_case(runs: Iterable[MillingRun]) -> Dict[int, List[MillingRun]]:
    grouped: Dict[int, List[MillingRun]] = {}
    for run in runs:
        grouped.setdefault(run.case_id, []).append(run)
    for case_runs in grouped.values():
        case_runs.sort(key=lambda item: item.run_id)
    return grouped


def feature_names(
    channels: Sequence[str] = DEFAULT_CHANNELS,
    include_process_features: bool = True,
) -> List[str]:
    stat_names = [
        "mean",
        "std",
        "rms",
        "peak",
        "ptp",
        "abs_mean",
        "skew",
        "kurtosis",
        "shape_factor",
        "crest_factor",
        "impulse_factor",
        "clearance_factor",
        "energy",
    ]
    names = [f"{channel}_{stat}" for channel in channels for stat in stat_names]
    if include_process_features:
        names.extend(["DOC", "feed", "material"])
    return names
