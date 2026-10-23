"""训练入口：使用 PyTorch Lightning 的 ``Trainer``。

用法::

    python -m catdog.train --model dnn --epochs 20
    python -m catdog.train --model cnn --epochs 20
    python -m catdog.train --model rnn --epochs 20
    python -m catdog.train --model cnn_attention --epochs 20
    python -m catdog.train --model vit --epochs 20
    python -m catdog.train --model dnn --fast-dev-run    # 1 个 batch 冒烟测试

"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import pytorch_lightning as pl

from catdog.data import CatDogDataModule
from catdog.models.CNN import CNN, CNNConfig
from catdog.models.CNN.trainer import build_trainer as build_cnn_trainer
from catdog.models.CNN_attention import CNNAttention, CNNAttentionConfig
from catdog.models.CNN_attention.trainer import (
    build_trainer as build_cnn_attention_trainer,
)
from catdog.models.DNN import DNN, DNNConfig
from catdog.models.DNN.trainer import build_trainer as build_dnn_trainer
from catdog.models.RNN import RNN, RNNConfig
from catdog.models.RNN.trainer import build_trainer as build_rnn_trainer
from catdog.models.ViT import ViT, ViTConfig
from catdog.models.ViT.trainer import build_trainer as build_vit_trainer

# 固定随机种子：保证权重初始化、数据打乱、增强、dropout 可复现。
SEED: int = 42

# 模型名 -> 对应的 Trainer 工厂
TRAINER_BUILDERS = {
    "dnn": build_dnn_trainer,
    "cnn": build_cnn_trainer,
    "rnn": build_rnn_trainer,
    "cnn_attention": build_cnn_attention_trainer,
    "vit": build_vit_trainer,
}

# 模型名 -> 配置类（用于取默认 epoch 数等）
CONFIG_CLASSES = {
    "dnn": DNNConfig,
    "cnn": CNNConfig,
    "rnn": RNNConfig,
    "cnn_attention": CNNAttentionConfig,
    "vit": ViTConfig,
}


def default_epochs(model_name: str) -> int:
    """取该模型 ``cfg.py`` 里推荐的 ``max_epochs``（命令行不传 ``--epochs`` 时用）。"""
    return CONFIG_CLASSES[model_name]().max_epochs


def build_dnn(
    num_workers: int = 0,
    stats_max_samples: Optional[int] = None,
    cfg: Optional[DNNConfig] = None,
) -> Tuple[pl.LightningModule, CatDogDataModule]:
    """根据 ``DNNConfig`` 构建模型与 DataModule（数据配置来自 ``cfg.data``）。

    ``cfg`` 为空时使用默认配置；调参脚本（``catdog.tune``）会传入自己的配置。
    """
    cfg = cfg or DNNConfig()
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
    cfg: Optional[CNNConfig] = None,
) -> Tuple[pl.LightningModule, CatDogDataModule]:
    """根据 ``CNNConfig`` 构建 ResNet-18 风格的 CNN 与 DataModule。

    ``cfg`` 为空时使用默认配置。
    """
    cfg = cfg or CNNConfig()
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


def build_rnn(
    num_workers: int = 0,
    stats_max_samples: Optional[int] = None,
    cfg: Optional[RNNConfig] = None,
) -> Tuple[pl.LightningModule, CatDogDataModule]:
    """根据 ``RNNConfig`` 构建 patch embedding + LSTM 模型与 DataModule。

    ``cfg`` 为空时使用默认配置。
    """
    cfg = cfg or RNNConfig()
    model = RNN(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        patch_size=cfg.patch_size,
        pic_size=cfg.pic_size,
        hidden_dim=cfg.hidden_dim,
        bottleneck_dim=cfg.bottleneck_dim,
        rnn_hidden=cfg.rnn_hidden,
        num_layers=cfg.num_layers,
        bidirectional=cfg.bidirectional,
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


def build_cnn_attention(
    num_workers: int = 0,
    stats_max_samples: Optional[int] = None,
    cfg: Optional[CNNAttentionConfig] = None,
) -> Tuple[pl.LightningModule, CatDogDataModule]:
    """根据 ``CNNAttentionConfig`` 构建 CNN + Attention 模型与 DataModule。

    ``cfg`` 为空时使用默认配置。
    """
    cfg = cfg or CNNAttentionConfig()
    model = CNNAttention(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_width=cfg.base_width,
        layers=cfg.layers,
        fnn_hidden=cfg.fnn_hidden,
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


def build_vit(
    num_workers: int = 0,
    stats_max_samples: Optional[int] = None,
    cfg: Optional[ViTConfig] = None,
) -> Tuple[pl.LightningModule, CatDogDataModule]:
    """根据 ``ViTConfig`` 构建 Vision Transformer 与 DataModule。

    ``cfg`` 为空时使用默认配置。预处理与 DNN / CNN / RNN 共用 ``image_preset()``。
    """
    cfg = cfg or ViTConfig()
    model = ViT(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        attn_dropout=cfg.attn_dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_frac=cfg.warmup_frac,
        min_lr_ratio=cfg.min_lr_ratio,
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
    cfg: Optional[
        DNNConfig | CNNConfig | RNNConfig | CNNAttentionConfig
    ] = None,
) -> Tuple[pl.LightningModule, CatDogDataModule]:
    """模型注册表。``cfg`` 为空时各模型使用自己 cfg.py 里的默认配置。"""
    if model_name == "dnn":
        return build_dnn(
            num_workers=num_workers, stats_max_samples=stats_max_samples, cfg=cfg
        )
    if model_name == "cnn":
        return build_cnn(
            num_workers=num_workers, stats_max_samples=stats_max_samples, cfg=cfg
        )
    if model_name == "rnn":
        return build_rnn(
            num_workers=num_workers, stats_max_samples=stats_max_samples, cfg=cfg
        )
    if model_name == "cnn_attention":
        return build_cnn_attention(
            num_workers=num_workers, stats_max_samples=stats_max_samples, cfg=cfg
        )
    if model_name == "vit":
        return build_vit(
            num_workers=num_workers, stats_max_samples=stats_max_samples, cfg=cfg
        )
    raise ValueError(f"unknown model: {model_name}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cat vs Dog training with PyTorch Lightning"
    )
    parser.add_argument(
        "--model",
        default="dnn",
        choices=["dnn", "cnn", "rnn", "cnn_attention", "vit"],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数；不传则用该模型 cfg.py 里的 max_epochs",
    )
    parser.add_argument("--accelerator", default="auto", help="auto / gpu / cpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stats-max-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument(
        "--monitor",
        default="val_loss",
        choices=["val_loss", "val_acc"],
        help=(
            "ModelCheckpoint 监控的指标。默认 val_loss（mode=min）；"
            "传 val_acc 时按准确率挑最优权重（mode=max）——"
            "训练里 val_loss 最低点和 val_acc 峰值常常不在同一轮，关心准确率时用这个。"
            "注意 EarlyStopping 固定看 val_loss，不跟随此参数：val_acc 早期抖动大"
            "（patience 内 ±2 pt），用它早停会提前终止训练。"
        ),
    )
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
    epochs = args.epochs if args.epochs is not None else default_epochs(args.model)
    print(f"[train] model={args.model} epochs={epochs} monitor={args.monitor}")
    trainer = trainer_builder(
        max_epochs=epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        out_dir=args.out_dir,
        monitor=args.monitor,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
