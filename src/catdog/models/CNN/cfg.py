from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DataConfig, image_preset


@dataclass
class CNNConfig:
    """CNN的模型配置。

    预处理与其余固定尺寸模型共用 ``image_preset()``（第三轮起训练管线含
    随机旋转 ±15°，在 Normalize 之后），超参与 ``CNN.__init__`` 一一对应。

# 第三轮调参（image_preset 增加 RandomRotate ±15° 之后重新搜索）
# 最优 trial：val_acc=91.20%，收敛 101 轮
# 注意 lr 与训练轮数是强耦合的：大 lr 需要更多轮才能收敛，但收敛后精度更高。
    """

    data: DataConfig = field(
        default_factory=lambda: image_preset(batch_size=32)
    )

    in_channels: int = 3
    num_classes: int = 2
    layers: tuple[int, int, int, int] = (2, 2, 2, 2)  # 每个 stage 的 BasicBlock 数
    base_width: int = 64
    dropout: float = 0.078
    lr: float = 0.00404
    weight_decay: float = 2.3e-05
    max_epochs: int = 300   # 只是上限：实际由 EarlyStopping(val_loss, patience=10) 终止
