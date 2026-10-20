from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import ATTENTION_BATCH_SIZE, DataConfig
from catdog.data.transforms import TransformConfig
from catdog.data.types import TransformType


def _attention_data(batch_size: int = ATTENTION_BATCH_SIZE) -> DataConfig:
    """CNN_attention 的专用预处理：保留原图尺寸，**不缩放、不裁剪**。

    CNN 会在原图上产生数量不定的特征向量（token），再由 attention 聚合。
    因此这里只有 ``Normalize``；变尺寸 batch 由补零 + mask 处理。
    """
    return DataConfig(
        variable_size=True,
        batch_size=batch_size,
        train=[TransformConfig(TransformType.Normalize)],
        val=[TransformConfig(TransformType.Normalize)],
    )


@dataclass
class CNNAttentionConfig:
    """CNN + Attention 的模型配置。

    ``data`` 使用变尺寸、仅归一化的预处理；其余超参待定。
    """

    data: DataConfig = field(default_factory=_attention_data)
    # TODO: 模型超参数（d_model / num_heads / num_layers 等）
