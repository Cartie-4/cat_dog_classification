from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DataConfig, image_preset


@dataclass
class CNNConfig:
    """CNN 的模型配置。

    预处理与其余固定尺寸模型共用 ``image_preset()``；模型超参待定。
    """

    data: DataConfig = field(default_factory=image_preset)
    # TODO: 模型超参数（通道数 / 卷积层数 / lr 等）
