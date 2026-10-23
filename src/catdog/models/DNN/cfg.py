from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DataConfig, image_preset, DEFAULT_IMAGE_SIZE


@dataclass
class DNNConfig:
    """DNN 的模型配置。

    预处理与其余固定尺寸模型共用 ``image_preset()``（第三轮起训练管线含
    随机旋转 ±15°，在 Normalize 之后），超参数与 ``DNN.__init__`` 一一对应。

# 第三轮调参（image_preset 增加 RandomRotate ±15° 之后重新搜索）
# 最优 trial：val_acc=69.00%，收敛 63 轮
# lr / wd / dropout / hidden_dim / bottleneck_dim / batch 全部来自本轮搜索。
    """

    data: DataConfig = field(
        default_factory=lambda: image_preset(batch_size=32)
    )

    in_dim: int = DEFAULT_IMAGE_SIZE * DEFAULT_IMAGE_SIZE * 3
    hidden_dim: int = 512
    bottleneck_dim: int = 128
    out_dim: int = 2
    dropout: float = 0.273
    lr: float = 1.79e-05
    weight_decay: float = 0.00505
    max_epochs: int = 300   # 只是上限：实际由 EarlyStopping(val_loss, patience=10) 终止
