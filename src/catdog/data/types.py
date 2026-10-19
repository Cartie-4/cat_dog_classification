import os
from enum import Enum
from pathlib import Path
from typing import TypeAlias

import torch
from jaxtyping import Float

# 数据根目录：优先环境变量 CATDOG_DATA_DIR，其次当前工作目录下的 data/。
# 说明：安装为库后（非 editable），__file__ 位于 site-packages，不能再据此
# 反推项目根目录，因此改为可配置。训练入口也可直接给 dataset/datamodule 传 root。
DATA_ROOT = Path(os.environ.get("CATDOG_DATA_DIR", Path.cwd() / "data"))

DATASET_DIR: dict[str, str] = {
    "kaggle": str(DATA_ROOT),
}

Pixel1: TypeAlias = Float[torch.Tensor, "H W C"]
PixelB: TypeAlias = Float[torch.Tensor, "B H W C"]

class TransformType(Enum):
    Identity = "identity"
    Resize = "resize"
    RandomCrop = "random_crop"
    RandomFlip = "random_flip"
    RandomRotate = "random_rotate"
    CenterCrop = "center_crop"
    Normalize = "normalize"

