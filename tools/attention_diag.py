"""诊断：CNN_attention 为什么没有比 CNN 更好。

做四组对照实验（都用 val_acc 比较）：

1. ``padstats``  —— 只跑数据：统计变尺寸 batch 的补零占比（batch 8 / 32）。
2. ``exp`` —— 用指定的数据管线 + 结构开关跑完整训练，输出逐 epoch 曲线：
   - ``--data attn``    : 模型原配预处理（原图尺寸，翻转+旋转）
   - ``--data img192``  : CNN 那套预处理（缩放到 192 -> 随机裁剪 192，翻转），
                          变量尺寸 collate 保留（补零退化为无操作），这样 mask 仍在
   - ``--head attn``    : 原始单 query 注意力聚合
   - ``--head gap``     : 换成 masked 全局平均池化（其余完全相同）

用法::

    python outputs/experiments/attention_diag.py --exp padstats
    python outputs/experiments/attention_diag.py --exp run --data img192 --epochs 10 --batch-size 32
    python outputs/experiments/attention_diag.py --exp run --data attn --epochs 20 --batch-size 8
    python outputs/experiments/attention_diag.py --exp run --data img192 --epochs 10 --head gap
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from catdog.data.cfg import DataConfig
from catdog.data.datamodule import CatDogDataModule
from catdog.data.dataset import variable_collate_fn
from catdog.data.transforms import TransformConfig
from catdog.data.types import TransformType
from catdog.models.CNN_attention import CNNAttention
from catdog.models.CNN_attention.cfg import _attention_data
from catdog.tune import _MetricsRecorder, _build_tuning_trainer

# 调优后的 CNN_attention 超参（见 models/CNN_attention/cfg.py）
TUNED = dict(
    lr=1.15e-4,
    weight_decay=7.49e-6,
    dropout=0.202,
    fnn_hidden=128,
    base_width=64,
)


def image192_data(batch_size: int) -> DataConfig:
    """CNN 那套预处理 + 变尺寸 collate（尺寸统一后补零是空操作，但 mask 仍存在）。"""
    return DataConfig(
        variable_size=True,
        batch_size=batch_size,
        train=[
            TransformConfig(TransformType.Resize, {"size": 192}),
            TransformConfig(TransformType.RandomCrop, {"size": 192}),
            TransformConfig(TransformType.RandomFlip, {"p_horizontal": 0.5}),
            TransformConfig(TransformType.Normalize),
        ],
        val=[
            TransformConfig(TransformType.Resize, {"size": 192}),
            TransformConfig(TransformType.CenterCrop, {"size": 192}),
            TransformConfig(TransformType.Normalize),
        ],
    )


class CNNAttentionGAP(CNNAttention):
    """把注意力聚合换成 masked 全局平均池化，其余结构完全一致。"""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        feat = self.trunk(x)
        b, d, hf, wf = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, d)

        token_mask = F.adaptive_max_pool2d(
            mask.float().unsqueeze(1), (hf, wf)
        ).squeeze(1)
        token_mask = token_mask.flatten(1) > 0
        weights = token_mask.float().unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.fnn(self.norm(pooled))


def padding_stats() -> None:
    """统计变尺寸 collate 下的补零比例。"""
    cfg = _attention_data(batch_size=8)
    dm = CatDogDataModule(cfg, num_workers=0)
    dm.setup("validate")

    print("=== 变尺寸 batch 的补零占比（验证集 499 张）===")
    for bs in (8, 16, 32):
        loader = DataLoader(
            dm._val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            collate_fn=variable_collate_fn,
        )
        fracs, shapes = [], []
        for images, mask, _ in loader:
            fracs.append(1.0 - float(mask.float().mean()))
            shapes.append((int(images.shape[1]), int(images.shape[2])))
        mean_pad = sum(fracs) / len(fracs)
        hw = [h * w for h, w in shapes]
        print(
            f"batch={bs:<3} 补零占比 {mean_pad * 100:5.1f}%  "
            f"batch 内最大画布 {max(hw) / 1e3:6.1f}k 像素  "
            f"平均 {sum(hw) / len(hw) / 1e3:6.1f}k 像素"
        )
    print(
        "对比：固定 192x192 时每张图是 36.9k 像素；原图均值约 "
        f"{_mean_area():.1f}k 像素"
    )


def _mean_area() -> float:
    from catdog.data.dataset import CatDogDataset
    from PIL import Image

    ds = CatDogDataset("val", transform=None)
    total = 0
    for path, _ in ds.samples:
        with Image.open(path) as im:
            total += im.width * im.height
    return total / len(ds.samples) / 1e3


def run_exp(args) -> dict:
    pl.seed_everything(args.seed, workers=True)

    cfg_data = (
        _attention_data(args.batch_size)
        if args.data == "attn"
        else image192_data(args.batch_size)
    )
    model_kwargs = dict(
        in_channels=3,
        num_classes=2,
        base_width=args.base_width,
        fnn_hidden=args.fnn_hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    cls = CNNAttentionGAP if args.head == "gap" else CNNAttention
    model = cls(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())

    dm = CatDogDataModule(cfg_data, num_workers=0)
    recorder = _MetricsRecorder()
    trainer = _build_tuning_trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        callbacks=[recorder],
    )

    started = time.perf_counter()
    trainer.fit(model, datamodule=dm)
    elapsed = time.perf_counter() - started

    curve = [round(r["val_acc"], 4) for r in recorder.records]
    best = max(curve) if curve else None
    print(
        f"[{args.data}/{args.head}] epochs={args.epochs} bs={args.batch_size} "
        f"params={n_params / 1e6:.2f}M best_acc={best:.4f} sec={elapsed:.0f} "
        f"({elapsed / max(1, len(curve)):.1f}s/epoch)"
    )
    print(f"  curve {curve}")
    del trainer, model, dm
    torch.cuda.empty_cache()
    return {
        "data": args.data,
        "head": args.head,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "params_m": round(n_params / 1e6, 2),
        "best_acc": best,
        "curve": curve,
        "seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN_attention 诊断实验")
    parser.add_argument("--exp", choices=["padstats", "run"], default="run")
    parser.add_argument("--data", choices=["attn", "img192"], default="img192")
    parser.add_argument("--head", choices=["attn", "gap"], default="attn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=TUNED["lr"])
    parser.add_argument("--weight-decay", type=float, default=TUNED["weight_decay"])
    parser.add_argument("--dropout", type=float, default=TUNED["dropout"])
    parser.add_argument("--fnn-hidden", type=int, default=TUNED["fnn_hidden"])
    parser.add_argument("--base-width", type=int, default=TUNED["base_width"])
    parser.add_argument("--out", default="outputs/experiments/attention_diag.jsonl")
    args = parser.parse_args()

    if args.exp == "padstats":
        padding_stats()
        return

    result = run_exp(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
