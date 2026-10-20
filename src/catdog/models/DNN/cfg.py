from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DataConfig, image_preset, DEFAULT_IMAGE_SIZE


@dataclass
class DNNConfig:
    """DNN 的模型配置。

    预处理与其余固定尺寸模型共用 ``image_preset()``；
    超参数与 ``DNN.__init__`` 的参数一一对应。
    """

    data: DataConfig = field(default_factory=image_preset)

    in_dim: int = DEFAULT_IMAGE_SIZE * DEFAULT_IMAGE_SIZE * 3
    hidden_dim: int = 256
    bottleneck_dim: int = 128
    out_dim: int = 2
    dropout: float = 0.1
    lr: float = 1e-5
    max_epochs: int = 50
