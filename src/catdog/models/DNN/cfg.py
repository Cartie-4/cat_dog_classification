from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DataConfig, image_preset, DEFAULT_IMAGE_SIZE


@dataclass
class DNNConfig:
    """DNN 的模型配置。

    预处理与其余固定尺寸模型共用 ``image_preset()``；超参数与 ``DNN.__init__``
    的参数一一对应。

    调参结论（第二轮：每组超参训练到**自己的早停点**再比较）
    -------------------------------------------------------
    - 最优：lr 1.79e-5、hidden_dim 512、bottleneck_dim 128、dropout 0.273、
      weight_decay 5.05e-3，**66.80%**（约 28 轮到早停）；
    - lr 越大越差这条规律两轮都成立：lr 2.13e-4 -> 60.60%，lr 1.01e-3 -> 54.80%，
      lr 9.29e-3（CNN 那一档）在这里直接崩到 50% 以下；
    - DNN 的天花板就在 67% 左右：输入是 192*192*3=110592 维的原始像素，
      首层就吃掉 28.3M 参数，架构本身不适配图像。
    """

    data: DataConfig = field(default_factory=image_preset)

    in_dim: int = DEFAULT_IMAGE_SIZE * DEFAULT_IMAGE_SIZE * 3
    hidden_dim: int = 512     # 调参：256 -> 512
    bottleneck_dim: int = 128
    out_dim: int = 2
    dropout: float = 0.273    # 调参：0.1 -> 0.273
    lr: float = 1.79e-5       # 调参：5e-5 -> 1.79e-5
    weight_decay: float = 5.05e-3   # 调参：1e-2 -> 5.05e-3
    max_epochs: int = 300   # 只是上限：实际由 EarlyStopping(val_loss, patience=10) 终止
