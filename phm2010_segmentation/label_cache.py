import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .pseudo_label import PseudoLabelConfig


def config_to_dict(config: PseudoLabelConfig) -> dict[str, Any]:
    return {
        "smooth_window": int(config.smooth_window),
        "active_threshold": float(config.active_threshold),
        "inactive_threshold": float(config.inactive_threshold),
        "transition_ratio": float(config.transition_ratio),
        "min_transition_points": int(config.min_transition_points),
        "min_active_points": int(config.min_active_points),
        "min_cut_ratio": float(config.min_cut_ratio),
        "max_gap_ratio": float(config.max_gap_ratio),
        "edge_margin_ratio": float(config.edge_margin_ratio),
    }


def config_fingerprint(config: PseudoLabelConfig) -> str:
    payload = json.dumps(config_to_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def cache_path_for_cut(
    cut_file: str | Path,
    data_root: str | Path,
    label_cache_dir: str | Path,
) -> Path:
    cut_file = Path(cut_file)
    data_root = Path(data_root)
    label_cache_dir = Path(label_cache_dir)
    rel = cut_file.resolve().relative_to(data_root.resolve())
    safe_name = "__".join(rel.with_suffix("").parts) + ".npz"
    return label_cache_dir / safe_name


def save_label_cache(
    cache_path: str | Path,
    labels: np.ndarray,
    score: np.ndarray,
    metadata: dict[str, Any],
    source_file: str | Path,
    data_root: str | Path,
    config: PseudoLabelConfig,
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rel_source = str(Path(source_file).resolve().relative_to(Path(data_root).resolve()))
    np.savez_compressed(
        cache_path,
        labels=np.asarray(labels, dtype=np.int64),
        score=np.asarray(score, dtype=np.float32),
        metadata_json=json.dumps(metadata, sort_keys=True),
        source_relpath=rel_source,
        config_json=json.dumps(config_to_dict(config), sort_keys=True),
        config_hash=config_fingerprint(config),
    )


def load_label_cache(
    cache_path: str | Path,
    expected_config: PseudoLabelConfig | None = None,
    strict_config: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        labels = cache["labels"].astype(np.int64)
        score = cache["score"].astype(np.float32)
        metadata = json.loads(str(cache["metadata_json"].item()))
        if expected_config is not None and strict_config:
            expected_hash = config_fingerprint(expected_config)
            actual_hash = str(cache["config_hash"].item())
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Label cache config mismatch for {cache_path}. "
                    f"expected={expected_hash}, actual={actual_hash}. "
                    "Rebuild labels with build_label_cache.py or pass matching pseudo-label arguments."
                )
    return labels, score, metadata
