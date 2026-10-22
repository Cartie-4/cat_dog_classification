from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DEFAULT_IMAGE_SIZE, DataConfig, image_preset


@dataclass
class RNNConfig:
    """RNN（patch embedding + LSTM）的模型配置。

    预处理与其余固定尺寸模型共用 ``image_preset()``；超参与 ``RNN.__init__``
    的参数一一对应。

    调参结论（第二轮：每组超参训练到**自己的早停点**再比较）
    -------------------------------------------------------
    - 最优：lr 9.18e-5、rnn_hidden 128、hidden_dim 256、num_layers 1、batch 64、
      dropout 0.013、weight_decay 1.05e-4，**72.80%**（约 77 轮到早停）；
    - 第一轮的冠军（lr 1.1e-3）在给足预算后只有 69.60%，即本轮把 lr **降了约 12 倍**更好；
    - 低 lr 要更多轮：lr 6.65e-5 跑了 144 轮才停（71.20%），lr 9.18e-5 跑 77 轮（72.80%）。
      这就是"epoch 数要跟着 lr 走"的体现。
    """

    data: DataConfig = field(
        default_factory=lambda: image_preset(batch_size=64)
    )

    in_channels: int = 3
    num_classes: int = 2
    patch_size: int = 16
    pic_size: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    hidden_dim: int = 256        # patch embedding 输出通道
    bottleneck_dim: int = 128    # patch embedding 的中间通道
    rnn_hidden: int = 128        # 调参：256 -> 128
    num_layers: int = 1          # 调参：2 -> 1
    bidirectional: bool = True
    dropout: float = 0.013       # 调参：0.114 -> 0.013
    lr: float = 9.18e-5          # 调参：1.1e-3 -> 9.18e-5（需要更多轮，约 77 轮）
    weight_decay: float = 1.05e-4   # 调参：1.53e-6 -> 1.05e-4
    max_epochs: int = 300   # 只是上限：实际由 EarlyStopping(val_loss, patience=10) 终止
