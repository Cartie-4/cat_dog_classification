"""示例预处理模块的导出。

方便这样使用::

    from catdog.example_data import CatDogDataModule, image_preset, attention_preset

.. note::
   本包（``catdog.example_data``）**仅作为自包含示例**，把所有预处理代码集中
   在一处方便阅读。

   正式项目中，「某个模型使用哪套预处理」应写在对应模型的 ``cfg.py`` 中
   （如 ``catdog/models/DNN/cfg.py``、``catdog/models/CNN_attention/cfg.py``），
   而不是放在数据层的通用配置里。``attention_preset`` 这类模型专属预设
   尤其应随模型移动。
"""

from .config import (
    ATTENTION_BATCH_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    MIN_SHORT_SIDE,
    DataConfig,
    attention_preset,
    image_preset,
)
from .datamodule import CatDogDataModule
from .dataset import CatDogDataset, variable_collate_fn
from .transforms import (
    Compose,
    Normalize,
    TransformConfig,
    build_pipeline,
    compute_mean_std,
)

__all__ = [
    "ATTENTION_BATCH_SIZE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_IMAGE_SIZE",
    "MIN_SHORT_SIDE",
    "DataConfig",
    "attention_preset",
    "image_preset",
    "CatDogDataModule",
    "CatDogDataset",
    "variable_collate_fn",
    "Compose",
    "Normalize",
    "TransformConfig",
    "build_pipeline",
    "compute_mean_std",
]
