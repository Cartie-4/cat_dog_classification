"""数据读取与预处理。

- ``types``：张量布局约定、``TransformType`` 枚举、数据根目录。
- ``transforms``：几何变换、``Normalize``、配置工厂、统计量。
- ``dataset``：``CatDogDataset`` 与变尺寸 collate。
- ``datamodule``：``CatDogDataModule``（LightningDataModule）。

常用入口::

    from catdog.data import CatDogDataModule, image_preset
"""

from .cfg import (
    ATTENTION_BATCH_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    MIN_SHORT_SIDE,
    DataConfig,
    image_preset,
)
from .dataset import (
    CLASSES,
    LABELS,
    CatDogDataset,
    make_collate_fn,
    variable_collate_fn,
)
from .datamodule import DEFAULT_STATS_PATH, CatDogDataModule
from .transforms import (
    CenterCrop,
    Compose,
    IdentityTransform,
    Normalize,
    RandomCrop,
    RandomFlip,
    RandomRotate,
    Resize,
    TransformConfig,
    build_pipeline,
    build_transform,
    compute_mean_std,
    pipeline_signature,
)
from .types import DATASET_DIR, Pixel1, PixelB, TransformType

__all__ = [
    # config
    "ATTENTION_BATCH_SIZE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_IMAGE_SIZE",
    "MIN_SHORT_SIDE",
    "DataConfig",
    "image_preset",
    # dataset / collate
    "CLASSES",
    "LABELS",
    "CatDogDataset",
    "make_collate_fn",
    "variable_collate_fn",
    # datamodule
    "DEFAULT_STATS_PATH",
    "CatDogDataModule",
    # transforms
    "CenterCrop",
    "Compose",
    "IdentityTransform",
    "Normalize",
    "RandomCrop",
    "RandomFlip",
    "RandomRotate",
    "Resize",
    "TransformConfig",
    "build_pipeline",
    "build_transform",
    "compute_mean_std",
    "pipeline_signature",
    # types
    "DATASET_DIR",
    "Pixel1",
    "PixelB",
    "TransformType",
]
