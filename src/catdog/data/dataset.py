from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import CenterCrop, Compose, RandomCrop, RandomFlip, Resize
from .types import DATASET_DIR

IMAGE_SIZE: int = 192
MIN_SHORT_SIDE: int = 96
LABELS: dict[str, int] = {"cat": 0, "dog": 1}
CLASSES: list[str] = ["cat", "dog"]


def build_transform(train: bool) -> Compose:
    if train:
        return Compose(
            [
                Resize(IMAGE_SIZE),
                RandomCrop(IMAGE_SIZE),
                RandomFlip(p_horizontal=0.5, p_vertical=0.0),
            ]
        )
    return Compose([Resize(IMAGE_SIZE), CenterCrop(IMAGE_SIZE)])


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
        self.min_short_side = min_short_side

        self.samples: list[Tuple[Path, int]] = []
        self.skipped: list[Tuple[str, int]] = []

        for path in sorted(self.dir.glob("*.jpg")):
            name = path.name.split(".")[0]
            if name not in LABELS:
                continue
            with Image.open(path) as im:
                short = min(im.width, im.height)
            if short < self.min_short_side:
                self.skipped.append((path.name, short))
                continue
            self.samples.append((path, LABELS[name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        with Image.open(path) as im:
            x = torch.from_numpy(np.array(im.convert("RGB"))).float()

        if self.transform is not None:
            x = self.transform(x)

        return x.contiguous(), torch.tensor(label, dtype=torch.long)


def build_dataloader(
    split: str,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    root: str = DATASET_DIR["kaggle"],
) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"
    dataset = CatDogDataset(
        split, root=root, transform=build_transform(train=shuffle)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
