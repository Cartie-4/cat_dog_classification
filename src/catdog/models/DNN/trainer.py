"""Trainer 配置（工厂函数）。

说明
----
- 这里只负责构建 ``pl.Trainer``，**不处理 batch 张量**。
  ``pl.Trainer`` 本身不会对输入做展平，因此「HWC -> (B, features)」的展平
  放在 ``DNN.forward`` 里完成。
- 用工厂函数而不是模块级 ``trainer = pl.Trainer(...)``，避免 import 时产生
  副作用（创建日志目录、初始化进程等）。

每次调用都会在 ``out_dir`` 下新建一个以日期时间命名的运行目录：

.. code-block:: text

    outputs/2026-10-20_15-04-05/
    ├── checkpoints/
    │   ├── epoch=03-val_loss=0.4123.ckpt   # 最优（val_loss 最低）
    │   └── last.ckpt                        # 最后一轮
    └── events.out.tfevents...               # TensorBoard 日志
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import tensorboard as tb


def build_trainer(
    max_epochs: int = 50,
    accelerator: str = "auto",
    devices: int = 1,
    out_dir: str = "outputs",
    fast_dev_run: bool = False,
) -> pl.Trainer:
    """构建训练用的 ``pl.Trainer``。

    - 运行目录：``<out_dir>/<YYYY-MM-DD_HH-MM-SS>/``
    - checkpoint：``save_top_k=1`` 保留最优，``save_last=True`` 另存最后一轮；
      文件名包含 epoch 与 val_loss。
    - logger：TensorBoard（``pip install tensorboard``）。
    """
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
                save_top_k=1,      # 只保留 val_loss 最低的一个
                save_last=True,    # 额外保存最后一轮 last.ckpt
                save_weights_only=True,  # 只存权重，不存优化器状态（文件更小）
                filename="DNN-epoch={epoch:02d}-val_loss={val_loss:.4f}",
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
