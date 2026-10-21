from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DataConfig, image_preset


@dataclass
class CNNConfig:
    """CNN的模型配置。

    预处理与其余固定尺寸模型共用 ``image_preset()``；超参与 ``CNN.__init__``
    的参数一一对应。
    """

    data: DataConfig = field(default_factory=image_preset)

    in_channels: int = 3
    num_classes: int = 2
    layers: tuple[int, int, int, int] = (2, 2, 2, 2)  # 每个 stage 的 BasicBlock 数
    base_width: int = 64
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
