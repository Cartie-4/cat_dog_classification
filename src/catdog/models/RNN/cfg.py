from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DataConfig, image_preset


@dataclass
class RNNConfig:
    """RNN 的模型配置。

    预处理与其余固定尺寸模型共用 ``image_preset()``；模型超参待定。
    """

    data: DataConfig = field(default_factory=image_preset)
    # TODO: 模型超参数（hidden_size / num_layers / 序列构造方式等）
