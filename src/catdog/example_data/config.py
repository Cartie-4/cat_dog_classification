"""数据相关的配置（示例）。

这里定义两类东西：
1. ``DataConfig``：描述一个模型在训练 / 验证时分别使用哪条预处理管线，
   以及批大小、是否变尺寸等。
2. 预设函数：``image_preset``（固定尺寸，供 DNN/CNN/RNN）和
   ``attention_preset``（原图尺寸，供 CNN_attention）。

注意：``DataConfig`` 里 **不出现 mean/std**，标准化所需统计量由
``compute_mean_std`` 在数据集上计算，并通过 ``Normalize`` 注入。

.. note::
   本文件位于 ``src/example_data``，**仅作为自包含示例**集中展示配置写法。

   在正式项目中，「某个模型用哪套预处理」应写在**该模型自己的 cfg.py** 里，
   而不是集中放在数据层：

   - ``src/models/DNN/cfg.py``::

         @dataclass
         class DNNConfig:
             data: DataConfig = field(default_factory=image_preset)

   - ``src/models/CNN_attention/cfg.py``::

         @dataclass
         class CNNAttentionConfig:
             data: DataConfig = field(default_factory=attention_preset)

   即：``DataConfig`` / ``image_preset`` 属于数据层，可复用；
   ``attention_preset`` 这种只在单个模型使用的预设，应随模型放进其 ``cfg.py``。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.types import TransformType

from .transforms import TransformConfig

# 默认模型统一裁剪到的正方形边长。192 可被 3 整除（192 / 3 = 64），
# 方便 CNN / RNN 共用同一套尺寸。
DEFAULT_IMAGE_SIZE: int = 192

# 构建数据集时忽略“最短边 < 96”的极小图（共 16 张）。
# 96 = 192 / 2，同样能被 3 整除。
MIN_SHORT_SIDE: int = 96

# 默认批大小；变尺寸（CNN_attention）时补零开销大，单独用小 batch。
DEFAULT_BATCH_SIZE: int = 32
ATTENTION_BATCH_SIZE: int = 8


@dataclass
class DataConfig:
    """一个模型的数据配置。

    Attributes
    ----------
    variable_size:
        False -> 所有样本形状一致，用 DataLoader 默认 collate。
        True  -> 保留原图尺寸，样本形状不一，用补零 + mask 的自定义 collate。
    batch_size:
        批大小。变尺寸模型建议设小一些（补零会放大显存占用）。
    train / val:
        训练 / 验证阶段的变换配置列表。``Normalize`` 只写枚举名，参数由
        统计量在构建管线时注入。
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

    每次调用都返回**新对象**，避免多个模型共享同一个可变 list。

    .. note::
       正式项目中，各模型的 ``cfg.py`` 通过
        ``data: DataConfig = field(default_factory=image_preset)`` 复用它。
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
