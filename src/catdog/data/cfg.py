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
    variable_size : bool
        是否保留原图尺寸（True 时使用变尺寸 collate：补零 + mask）。
    batch_size : int
        批大小。
    train : list[TransformConfig]
        训练集的预处理管线。
    val : list[TransformConfig]
        验证集的预处理管线。
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