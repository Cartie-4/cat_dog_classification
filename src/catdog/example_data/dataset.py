"""数据集与批处理（示例）。

包含三个部分：
1. ``CatDogDataset``：从 ``data/train`` / ``data/val`` 读取图片，过滤极小图，
   应用预处理管线，返回 ``(HWC 张量, label)``。
2. ``variable_collate_fn``：处理“变尺寸”样本的拼 batch —— 补零到 batch 内
   最大 H/W，并返回逐像素的有效性 mask。
3. ``make_collate_fn``：根据配置选择默认 collate 还是变尺寸 collate。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..data.types import DATASET_DIR

from .config import MIN_SHORT_SIDE

# 文件名形如 ``cat.123.jpg`` / ``dog.7.jpg``，前缀即类别。
LABELS: dict[str, int] = {"cat": 0, "dog": 1}
CLASSES: list[str] = ["cat", "dog"]


class CatDogDataset(Dataset):
    """猫狗二分类数据集。

    参数
    ----
    split:
        ``"train"`` 或 ``"val"``，对应 ``data/train`` / ``data/val``。
    transform:
        单个样本的预处理管线（``Compose``），输入/输出均为 HWC 张量。
    root:
        数据根目录，默认取项目里 ``DATASET_DIR["kaggle"]``。
    min_short_side:
        最短边小于该值的图片会被跳过（默认 96）。

    返回
    ----
    ``(x, label)``：
    - ``x``：``(H, W, 3)`` 的 float32 张量，取值范围 ``[0, 255]``。
      注意这里 **不做 /255**；归一化在 ``Normalize`` 里用数据集统计量完成。
    - ``label``：``int64`` 标量（cat=0, dog=1）。
    """

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

        # 记录被保留 / 被跳过的样本，便于审计和复现。
        self.samples: list[Tuple[Path, int]] = []
        self.skipped: list[Tuple[str, int]] = []

        for path in sorted(self.dir.glob("*.jpg")):
            name = path.name.split(".")[0]          # "cat.123.jpg" -> "cat"
            if name not in LABELS:
                continue

            with Image.open(path) as im:
                short = min(im.width, im.height)

            # 过滤极小图：放大后过于模糊，且对训练帮助不大。
            if short < min_short_side:
                self.skipped.append((path.name, short))
                continue

            self.samples.append((path, LABELS[name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]

        with Image.open(path) as im:
            # np.array(...) 会复制一份可写数组，避免 PyTorch 的只读警告。
            array = np.array(im.convert("RGB"))     # (H, W, 3), uint8
        x = torch.from_numpy(array).float()          # -> float32, [0, 255], HWC

        if self.transform is not None:
            x = self.transform(x)

        return x.contiguous(), torch.tensor(label, dtype=torch.long)


def variable_collate_fn(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """变尺寸样本的 collate：补零到 batch 内最大尺寸 + mask。

    参数
    ----
    batch:
        若干 ``(x, label)``，其中 ``x`` 是 ``(H_i, W_i, 3)``，各样本 H/W 可能不同。

    返回
    ----
    ``(images, mask, labels)``：
    - ``images``：``(B, Hmax, Wmax, 3)``，小图在右侧 / 下方补 0。
      由于 ``Normalize`` 之后 0 约等于“均值色”，补零本身是中性的；
      真正保证正确性的是 mask。
    - ``mask``：``(B, Hmax, Wmax)`` 的 bool，``True`` 表示原始图像像素。
    - ``labels``：``(B,)`` int64。

    模型侧用法：
        把 mask 按 CNN 的下采样倍数对齐到特征图尺寸，得到 token 级 mask，
        attention 在 softmax 前对无效 key 置 ``-inf``，从而忽略 padding。
    """
    xs, ys = zip(*batch)

    hmax = max(x.shape[0] for x in xs)
    wmax = max(x.shape[1] for x in xs)
    channels = xs[0].shape[-1]

    # 预分配“整批最大值”的容器，再逐张把有效区域拷贝进去。
    images = xs[0].new_zeros((len(xs), hmax, wmax, channels))
    mask = torch.zeros((len(xs), hmax, wmax), dtype=torch.bool)

    for i, x in enumerate(xs):
        h, w = x.shape[0], x.shape[1]
        images[i, :h, :w] = x
        mask[i, :h, :w] = True

    return images, mask, torch.stack(ys)


def make_collate_fn(variable_size: bool) -> Optional[Callable]:
    """根据是否变尺寸，返回对应 collate；固定尺寸时返回 None（用默认行为）。"""
    return variable_collate_fn if variable_size else None
