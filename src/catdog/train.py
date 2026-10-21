"""训练入口：使用 PyTorch Lightning 的 ``Trainer``。

用法::

    python -m catdog.train --model dnn --epochs 20
    python -m catdog.train --model dnn --fast-dev-run    # 1 个 batch 冒烟测试

"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import pytorch_lightning as pl

from catdog.data import CatDogDataModule
from catdog.models.CNN import CNN, CNNConfig
from catdog.models.CNN.trainer import build_trainer as build_cnn_trainer
from catdog.models.DNN import DNN, DNNConfig
from catdog.models.DNN.trainer import build_trainer as build_dnn_trainer

# 固定随机种子：保证权重初始化、数据打乱、增强、dropout 可复现。
SEED: int = 42

# 模型名 -> 对应的 Trainer 工厂
TRAINER_BUILDERS = {
    "dnn": build_dnn_trainer,
    "cnn": build_cnn_trainer,
}


def build_dnn(
    num_workers: int = 0,
    stats_max_samples: Optional[int] = None,
) -> Tuple[pl.LightningModule, CatDogDataModule]:
    """根据 ``DNNConfig`` 构建模型与 DataModule（数据配置来自 ``cfg.data``）。"""
    cfg = DNNConfig()
    model = DNN(
        in_dim=cfg.in_dim,
        hidden_dim=cfg.hidden_dim,
        bottleneck_dim=cfg.bottleneck_dim,
        out_dim=cfg.out_dim,
        dropout=cfg.dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    dm = CatDogDataModule(
        cfg.data,
        num_workers=num_workers,
        stats_max_samples=stats_max_samples,
    )
    return model, dm


def build_cnn(
    num_workers: int = 0,
    stats_max_samples: Optional[int] = None,
) -> Tuple[pl.LightningModule, CatDogDataModule]:
    """根据 ``CNNConfig`` 构建 ResNet-18 风格的 CNN 与 DataModule。"""
    cfg = CNNConfig()
    model = CNN(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=cfg.layers,
        base_width=cfg.base_width,
        dropout=cfg.dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    dm = CatDogDataModule(
        cfg.data,
        num_workers=num_workers,
        stats_max_samples=stats_max_samples,
    )
    return model, dm


def build(
    model_name: str,
    num_workers: int = 0,
    stats_max_samples: Optional[int] = None,
) -> Tuple[pl.LightningModule, CatDogDataModule]:
    """模型注册表：以后实现 RNN 时在这里扩展。"""
    if model_name == "dnn":
        return build_dnn(num_workers=num_workers, stats_max_samples=stats_max_samples)
    if model_name == "cnn":
        return build_cnn(num_workers=num_workers, stats_max_samples=stats_max_samples)
    if model_name == "rnn":
        raise NotImplementedError("rnn 模型尚未实现")
    raise ValueError(f"unknown model: {model_name}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cat vs Dog training with PyTorch Lightning"
    )
    parser.add_argument("--model", default="dnn", choices=["dnn", "cnn", "rnn"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--accelerator", default="auto", help="auto / gpu / cpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stats-max-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="只跑 1 个 train/val batch，用于冒烟测试",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # 必须在建模型 / DataModule 之前设置，才能固定初始化和数据顺序。
    # workers=True 同时给 DataLoader 的子进程播种（RandomCrop/RandomFlip 用的是 Python random）。
    pl.seed_everything(SEED, workers=True)

    model, dm = build(
        args.model,
        num_workers=args.num_workers,
        stats_max_samples=args.stats_max_samples,
    )

    trainer_builder = TRAINER_BUILDERS[args.model]
    trainer = trainer_builder(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        out_dir=args.out_dir,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
