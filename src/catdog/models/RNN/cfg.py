from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DEFAULT_IMAGE_SIZE, DataConfig, image_preset


@dataclass
class RNNConfig:
    """RNN（patch embedding + LSTM）的模型配置。

    预处理与其余固定尺寸模型共用 ``image_preset()``；超参与 ``RNN.__init__``
    的参数一一对应。
    """

    data: DataConfig = field(default_factory=image_preset)

    in_channels: int = 3
    num_classes: int = 2
    patch_size: int = 16
    pic_size: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    hidden_dim: int = 256        # patch embedding 的输出通道
    bottleneck_dim: int = 128    # patch embedding 的中间通道
    rnn_hidden: int = 256
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
