from __future__ import annotations
from dataclasses import dataclass, field

from .types import TransformType
from .transforms import TransformConfig

DEFAULT_IMAGE_SIZE: int = 192
MIN_SHORT_SIDE: int = 96

DEFAULT_BATCH_SIZE: int = 32
ATTENTION_BATCH_SIZE: int = 8

@dataclass
class DataConfig:
    """一个模型的数据配置。

    Attributes
    ----------
    train : list[TransformConfig]
        训练集的预处理管线。
    val : list[TransformConfig]
        验证集的预处理管线。
    batch_size : int
        批大小。
    """
    variable_size: bool = False
    batch_size: int = DEFAULT_BATCH_SIZE
    train: list[TransformConfig] = field(default_factory=list)
    val: list[TransformConfig] = field(default_factory=list)


def image_preset(
    size: int = DEFAULT_IMAGE_SIZE, batch_size: int = DEFAULT_BATCH_SIZE
) -> DataConfig:
    """固定尺寸模型的默认预处理（DNN / CNN / RNN 共用）。

    训练：短边缩放到 size -> 随机裁剪 size -> 随机水平翻转 -> 标准化
    验证：短边缩放到 size -> 中心裁剪 size -> 标准化
    """
    return DataConfig(
        variable_size=False,
        batch_size=batch_size,
        train=[
            TransformConfig(TransformType.Resize, {"size": size}),
            TransformConfig(TransformType.RandomCrop, {"size": size}),
            TransformConfig(TransformType.RandomFlip, {"p_horizontal": 0.5}),
            TransformConfig(TransformType.Normalize),
        ],
        val=[
            TransformConfig(TransformType.Resize, {"size": size}),
            TransformConfig(TransformType.CenterCrop, {"size": size}),
            TransformConfig(TransformType.Normalize),
        ],
    )


def attention_preset(batch_size: int = ATTENTION_BATCH_SIZE) -> DataConfig:
    """CNN_attention 的专用预处理：保留原图尺寸，**不缩放、不裁剪**。

    CNN 会在原图上产生数量不定的特征向量（token），再由 attention 聚合。
    因此这里只有 ``Normalize``；变尺寸 batch 由补零 + mask 处理。

    .. note::
       这是 **CNN_attention 专属** 的预设。正式项目中它应移动到
       ``src/models/CNN_attention/cfg.py``，作为 ``CNNAttentionConfig.data``
       的默认值，而不是留在通用数据配置里。本示例为了让所有代码集中在一处
       才暂时放在这里。
    """
    return DataConfig(
        variable_size=True,
        batch_size=batch_size,
        train=[TransformConfig(TransformType.Normalize)],
        val=[TransformConfig(TransformType.Normalize)],
    )