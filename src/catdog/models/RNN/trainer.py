"""RNN 的 Trainer 配置（工厂函数）。

与 ``models/DNN/trainer.py`` / ``models/CNN/trainer.py`` 基本一致：
- 只负责构建 ``pl.Trainer``，不处理 batch 张量；
- 用工厂函数避免 import 时产生副作用；
- 每次运行在 ``out_dir`` 下新建日期时间目录。

.. code-block:: text

    outputs/2026-10-20_15-04-05/
    ├── checkpoints/
    │   ├── epoch=03-val_loss=0.4123.ckpt
    │   └── last.ckpt
    └── events.out.tfevents...
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import tensorboard as tb


def build_trainer(
    max_epochs: int = 50,
    accelerator: str = "auto",
    devices: int = 1,
    out_dir: str = "outputs",
    fast_dev_run: bool = False,
) -> pl.Trainer:
    """构建 RNN 训练用的 ``pl.Trainer``。"""
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(out_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"

    logger = tb.TensorBoardLogger(
        save_dir=out_dir,
        name=run_name,
        version="",
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
            filename="RNN_epoch={epoch:02d}-val_loss={val_loss:.4f}",
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        default_root_dir=str(run_dir),
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks,
        logger=logger,
    )
