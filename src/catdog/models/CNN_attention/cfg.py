from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import ATTENTION_BATCH_SIZE, DataConfig
from catdog.data.transforms import TransformConfig
from catdog.data.types import TransformType


def _attention_data(batch_size: int = ATTENTION_BATCH_SIZE) -> DataConfig:
    """CNN_attention 的专用预处理：保留原图尺寸，**不缩放、不裁剪**。

    - 训练：随机水平翻转 -> Normalize -> 随机旋转（±15°）
      （旋转放在 Normalize 之后，补出来的边界正好是均值色，接近中性）
    - 验证：只做 Normalize，保证确定性。

    变尺寸 batch 由补零 + mask 处理。
    """
    return DataConfig(
        variable_size=True,
        batch_size=batch_size,
        train=[
            TransformConfig(TransformType.RandomFlip, {"p_horizontal": 0.5}),
            TransformConfig(TransformType.Normalize),
            TransformConfig(TransformType.RandomRotate, {"degrees": 15.0}),
        ],
        val=[TransformConfig(TransformType.Normalize)],
    )


@dataclass
class CNNAttentionConfig:
    """CNN + Attention 的模型配置。

    - ``data``：变尺寸、仅归一化的预处理（训练时加翻转/旋转）；
    - 其余超参与 ``CNNAttention.__init__`` 的参数一一对应。
    """

    data: DataConfig = field(default_factory=_attention_data)

    in_channels: int = 3
    num_classes: int = 2
    base_width: int = 64
    layers: tuple[int, int, int, int] = (2, 2, 2, 2)
    fnn_hidden: int = 256
    dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 30
