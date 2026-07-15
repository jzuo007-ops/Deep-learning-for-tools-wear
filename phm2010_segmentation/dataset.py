import random
import csv
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .pseudo_label import PseudoLabelConfig, generate_three_class_labels
from .label_cache import cache_path_for_cut, load_label_cache


TOOLS = ("c1", "c2", "c3", "c4", "c5", "c6")


def normalize_relative_cut_path(value: str | Path) -> str:
    path = Path(str(value).strip().replace("\\", "/"))
    return "/".join(path.parts).lower()


def load_excluded_cut_paths(csv_files: Sequence[str | Path] | None) -> set[str]:
    excluded: set[str] = set()
    for csv_file in csv_files or []:
        if csv_file is None or str(csv_file).strip() == "":
            continue
        path = Path(csv_file)
        if not path.exists():
            raise FileNotFoundError(f"Exclude sample CSV not found: {path}")
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if "cut_file" not in (reader.fieldnames or []):
                raise ValueError(f"Exclude sample CSV must contain a cut_file column: {path}")
            for row in reader:
                cut_file = (row.get("cut_file") or "").strip()
                if cut_file:
                    excluded.add(normalize_relative_cut_path(cut_file))
    return excluded


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
        task: str = "three_class",
        excluded_cut_paths: Sequence[str] | set[str] | None = None,
        eval_mode: str = "center",
        train_sampling: str = "multi_position_random",
        train_windows_per_cut: int = 5,
        eval_windows_per_cut: int = 5,
    ):
        self.data_root = Path(data_root)
        self.tools = tuple(tools)
        self.crop_length = int(crop_length)
        self.train = train
        if task not in {"three_class", "binary"}:
            raise ValueError(f"Unknown segmentation task={task!r}; expected three_class or binary")
        self.task = task
        if eval_mode not in {"center", "boundary", "multi_position"}:
            raise ValueError(
                f"Unknown eval_mode={eval_mode!r}; expected center, boundary, or multi_position"
            )
        self.eval_mode = eval_mode
        if train_sampling not in {"random", "multi_position", "multi_position_random"}:
            raise ValueError(
                f"Unknown train_sampling={train_sampling!r}; "
                "expected random, multi_position, or multi_position_random"
            )
        self.train_sampling = train_sampling
        self.train_windows_per_cut = max(1, int(train_windows_per_cut))
        self.eval_windows_per_cut = max(1, int(eval_windows_per_cut))
        self.excluded_cut_paths = {
            normalize_relative_cut_path(path)
            for path in (excluded_cut_paths or [])
        }
        self.files = list_cut_files(
            self.data_root,
            tools=self.tools,
            max_cuts_per_tool=max_cuts_per_tool,
        )
        if self.excluded_cut_paths:
            self.files = [
                path for path in self.files
                if normalize_relative_cut_path(path.resolve().relative_to(self.data_root.resolve()))
                not in self.excluded_cut_paths
            ]
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

    def _map_labels_for_task(self, labels: np.ndarray) -> np.ndarray:
        if self.task == "three_class":
            return labels
        return np.where(labels == 2, 1, 0).astype(np.int64)

    def __len__(self):
        if self.train:
            return len(self.files) * self.train_windows_per_cut
        if self.eval_mode == "center":
            return len(self.files)
        if self.eval_mode == "multi_position":
            return len(self.files) * self.eval_windows_per_cut
        return len(self.files) * 3

    @staticmethod
    def _multi_position_start(n_points: int, crop_length: int, slot: int, slots: int) -> int:
        last_start = max(0, n_points - crop_length)
        if last_start == 0:
            return 0
        if slots <= 1:
            return last_start // 2
        positions = np.linspace(0, last_start, num=slots)
        return int(round(float(positions[int(slot) % slots])))

    def _crop_bounds(self, n_points: int, eval_slot: int = 0) -> tuple[int, int]:
        if n_points <= self.crop_length:
            return 0, n_points
        if self.train:
            if self.train_sampling == "random":
                start = random.randint(0, n_points - self.crop_length)
            else:
                start = self._multi_position_start(
                    n_points=n_points,
                    crop_length=self.crop_length,
                    slot=eval_slot,
                    slots=self.train_windows_per_cut,
                )
                if self.train_sampling == "multi_position_random":
                    jitter = max(1, self.crop_length // 4)
                    start += random.randint(-jitter, jitter)
                    start = min(max(start, 0), n_points - self.crop_length)
        elif self.eval_mode == "boundary":
            starts = [0, (n_points - self.crop_length) // 2, n_points - self.crop_length]
            start = starts[int(eval_slot) % len(starts)]
        elif self.eval_mode == "multi_position":
            start = self._multi_position_start(
                n_points=n_points,
                crop_length=self.crop_length,
                slot=eval_slot,
                slots=self.eval_windows_per_cut,
            )
        else:
            start = (n_points - self.crop_length) // 2
        return start, start + self.crop_length

    def __getitem__(self, index):
        eval_slot = 0
        if self.train:
            file_index = index // self.train_windows_per_cut
            eval_slot = index % self.train_windows_per_cut
        elif self.eval_mode == "center":
            file_index = index
        else:
            slots = self.eval_windows_per_cut if self.eval_mode == "multi_position" else 3
            file_index = index // slots
            eval_slot = index % slots
        path = self.files[file_index]
        data = read_cut_csv(path)
        labels = self._map_labels_for_task(self._load_labels(path, data))
        start, end = self._crop_bounds(len(data), eval_slot=eval_slot)
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
