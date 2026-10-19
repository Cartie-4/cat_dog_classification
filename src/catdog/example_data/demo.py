"""预处理示例的运行入口。

安装后（项目根目录执行 ``pip install -e .``）任选一种方式运行::

    python -m catdog.example_data.demo
    catdog-demo

会分别演示两种模型的预处理：
1. 固定尺寸模型（DNN / CNN / RNN）：Resize -> Crop -> Flip -> Normalize，
   得到形状一致的 ``(B, 192, 192, 3)``。
2. ``CNN_attention``：保留原图尺寸、只做 Normalize，得到“变尺寸 + mask”的
   ``(images, mask, labels)``。
"""

from __future__ import annotations

import torch

from .config import attention_preset, image_preset
from .datamodule import CatDogDataModule


def _describe_stats(stats):
    mean, std = stats
    fmt = lambda v: "[" + ", ".join(f"{x:.2f}" for x in v) + "]"
    return f"mean={fmt(mean)}  std={fmt(std)}"


def demo_fixed_size() -> None:
    print("=" * 70)
    print("1) 固定尺寸模型（image_preset）：Resize -> Crop -> Flip -> Normalize")
    print("=" * 70)

    # stats_max_samples 设小一点，示例首次运行更快；实际训练可设 None 用全量。
    # 正式项目中这里应改为模型配置驱动的预处理，例如：
    #     from catdog.models.DNN.cfg import DNNConfig
    #     dm = CatDogDataModule(DNNConfig().data, ...)
    dm = CatDogDataModule(image_preset(), stats_max_samples=200)
    dm.setup()

    print("train 样本数:", len(dm.train_dataloader().dataset))
    print("val   样本数:", len(dm.val_dataloader().dataset))
    print("统计量(训练集):", _describe_stats(dm.stats))

    images, labels = next(iter(dm.train_dataloader()))
    print("batch images:", tuple(images.shape), images.dtype)
    print("batch labels:", tuple(labels.shape), labels.tolist()[:8], "...")
    # 标准化后逐通道均值应接近 0。
    print("标准化后逐通道均值:", [round(v, 3) for v in images.mean(dim=(0, 1, 2)).tolist()])


def demo_variable_size() -> None:
    print()
    print("=" * 70)
    print("2) CNN_attention（attention_preset）：保留原图尺寸，仅 Normalize")
    print("=" * 70)

    dm = CatDogDataModule(attention_preset(), stats_max_samples=200)
    # 正式项目中，这个 config 应来自模型自己的 cfg.py：
    #     from catdog.models.CNN_attention.cfg import CNNAttentionConfig
    #     dm = CatDogDataModule(CNNAttentionConfig().data, ...)
    dm.setup()

    print("train 样本数:", len(dm.train_dataloader().dataset))
    print("val   样本数:", len(dm.val_dataloader().dataset))
    print("统计量(训练集):", _describe_stats(dm.stats))

    # 变尺寸 collate 返回三个对象：images, mask, labels。
    images, mask, labels = next(iter(dm.val_dataloader()))
    print("images:", tuple(images.shape), images.dtype, "  # (B, Hmax, Wmax, C)")
    print("mask  :", tuple(mask.shape), mask.dtype, "  # True=真实像素")
    print("labels:", tuple(labels.shape), labels.tolist()[:8], "...")
    print("每张图有效像素数:", mask.sum(dim=(1, 2)).tolist())
    print("注意：每张图 H/W 不同 -> mask 有效区域不同，可交给 attention 掩码。")


def main() -> None:
    demo_fixed_size()
    demo_variable_size()


if __name__ == "__main__":
    main()
