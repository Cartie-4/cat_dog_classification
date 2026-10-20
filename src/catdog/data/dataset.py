from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .types import DATASET_DIR
from .cfg import MIN_SHORT_SIDE


LABELS: dict[str, int] = {"cat": 0, "dog": 1}
CLASSES: list[str] = ["cat", "dog"]


class CatDogDataset(Dataset):
    def __init__(
        self,
        split: str,
        root: str = DATASET_DIR["kaggle"],
        transform: Optional[Callable] = None,
        min_short_side: int = MIN_SHORT_SIDE,
    ):
        self.split = split
        self.dir = Path(root) / split
        self.transform = transform

        self.samples: list[Tuple[Path, int]] = []
        self.skipped: list[Tuple[str, int]] = []

        for path in sorted(self.dir.glob("*.jpg")):
            name = path.name.split(".")[0]
            if name not in LABELS:
                continue

            with Image.open(path) as im:
                short = min(im.width, im.height)
            
            if short < min_short_side:
                self.skipped.append((path.name, short))
                continue
            
            self.samples.append((path, LABELS[name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]

        with Image.open(path) as im:
            array = np.array(im.convert("RGB"), dtype=np.uint8)
        x = torch.from_numpy(array).float()  # (H, W, 3), float32, [0, 255]

        if self.transform is not None:
            x = self.transform(x)

        return x.contiguous(), torch.tensor(label, dtype=torch.long)


def variable_collate_fn(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    用于 ``DataLoader`` 的 collate_fn，支持变长输入。

    参数
    ----
    batch: list of (x, label)
        - x: (H, W, 3) 的 float32 张量，取值范围 [0, 255]。
        - label: int64 标量。
    """
    xs, ys = zip(*batch)

    hmax = max(x.shape[0] for x in xs)
    wmax = max(x.shape[1] for x in xs)
    channels = xs[0].shape[-1]

    images = xs[0].new_zeros((len(xs), hmax, wmax, channels))
    mask = torch.zeros((len(xs), hmax, wmax), dtype=torch.bool)

    for i, x in enumerate(xs):
        h, w = x.shape[0], x.shape[1]
        images[i, :h, :w] = x
        mask[i, :h, :w] = True

    return images, mask, torch.stack(ys)


def make_collate_fn(variable_size: bool) -> Optional[Callable]:
    return variable_collate_fn if variable_size else None
