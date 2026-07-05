import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .pseudo_label import PseudoLabelConfig, generate_three_class_labels
from .label_cache import cache_path_for_cut, load_label_cache


TOOLS = ("c1", "c2", "c3", "c4", "c5", "c6")


def list_cut_files(
    data_root: str | Path,
    tools: Sequence[str] | None = None,
    max_cuts_per_tool: int | None = None,
) -> List[Path]:
    root = Path(data_root)
    selected_tools = tuple(tools or TOOLS)
    files: List[Path] = []
    for tool in selected_tools:
        tool_files = sorted((root / tool).glob("*.csv"))
        if max_cuts_per_tool is not None:
            tool_files = tool_files[:max_cuts_per_tool]
        files.extend(tool_files)
    return files


def read_cut_csv(path: str | Path) -> np.ndarray:
    return pd.read_csv(path, header=None).to_numpy(dtype=np.float32)


def normalize_window(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True)
    return ((window - mean) / np.maximum(std, 1e-6)).astype(np.float32)


class PHM2010SegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        tools: Sequence[str],
        crop_length: int = 8192,
        train: bool = True,
        max_cuts_per_tool: int | None = None,
        pseudo_label_config: PseudoLabelConfig | None = None,
        label_cache_dir: str | Path | None = None,
        require_label_cache: bool = False,
        strict_label_cache_config: bool = True,
    ):
        self.data_root = Path(data_root)
        self.tools = tuple(tools)
        self.crop_length = int(crop_length)
        self.train = train
        self.files = list_cut_files(
            self.data_root,
            tools=self.tools,
            max_cuts_per_tool=max_cuts_per_tool,
        )
        self.pseudo_label_config = pseudo_label_config or PseudoLabelConfig()
        self.label_cache_dir = Path(label_cache_dir) if label_cache_dir is not None else None
        self.require_label_cache = require_label_cache
        self.strict_label_cache_config = strict_label_cache_config
        if not self.files:
            raise ValueError(f"No cut files found in {self.data_root} for tools={self.tools}")

    def _load_labels(self, path: Path, data: np.ndarray) -> np.ndarray:
        if self.label_cache_dir is not None:
            cache_path = cache_path_for_cut(path, self.data_root, self.label_cache_dir)
            if cache_path.exists():
                labels, _, _ = load_label_cache(
                    cache_path,
                    expected_config=self.pseudo_label_config,
                    strict_config=self.strict_label_cache_config,
                )
                return labels
            if self.require_label_cache:
                raise FileNotFoundError(
                    f"Missing label cache for {path}: {cache_path}. "
                    "Run phm2010_segmentation/build_label_cache.py before training."
                )

        labels, _, _ = generate_three_class_labels(data, self.pseudo_label_config)
        return labels

    def __len__(self):
        return len(self.files)

    def _crop_bounds(self, n_points: int) -> tuple[int, int]:
        if n_points <= self.crop_length:
            return 0, n_points
        if self.train:
            start = random.randint(0, n_points - self.crop_length)
        else:
            start = (n_points - self.crop_length) // 2
        return start, start + self.crop_length

    def __getitem__(self, index):
        path = self.files[index]
        data = read_cut_csv(path)
        labels = self._load_labels(path, data)
        start, end = self._crop_bounds(len(data))
        window = data[start:end]
        target = labels[start:end]

        if len(window) < self.crop_length:
            pad = self.crop_length - len(window)
            window = np.pad(window, ((0, pad), (0, 0)), mode="edge")
            target = np.pad(target, (0, pad), mode="edge")

        window = normalize_window(window)
        signal = torch.from_numpy(window.T).float()
        label = torch.from_numpy(target.astype(np.int64)).long()
        return signal, label


def make_tool_split(test_tool: str, tools: Iterable[str] = TOOLS) -> tuple[list[str], list[str], list[str]]:
    tools = list(tools)
    if test_tool not in tools:
        raise ValueError(f"Unknown test_tool={test_tool!r}; expected one of {tools}")
    test_index = tools.index(test_tool)
    val_tool = tools[(test_index + 1) % len(tools)]
    train_tools = [tool for tool in tools if tool not in {test_tool, val_tool}]
    return train_tools, [val_tool], [test_tool]
