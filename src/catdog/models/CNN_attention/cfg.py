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

    调参结论（10 epoch，验证集 499 张）
    ----------------------------------
    - 这个模型对 lr 极其敏感：随机搜索里 lr >= 1.6e-3 的 4 组全部塌到
      50.1%（退化成只预测一个类），而 lr ~1.1e-4 能到 77%+；
    - ``fnn_hidden`` 256 -> 128、``dropout`` 0.1 -> 0.202、
      ``weight_decay`` 1e-4 -> 7.49e-6：78.36% -> 78.96%（10 epoch）；
    - ``base_width`` 保持 64：降到 32 虽然每 epoch 便宜约 15%、
      5 epoch 时更快到位（77.15%），但 10 epoch 后被反超（77.15% vs 78.96%）；
    - 保留原图尺寸（不缩放）意味着单个 epoch 约 33-38s，是四个模型里最贵的。
    """

    data: DataConfig = field(default_factory=_attention_data)

    in_channels: int = 3
    num_classes: int = 2
    base_width: int = 64
    layers: tuple[int, int, int, int] = (2, 2, 2, 2)
    fnn_hidden: int = 128       # 调参：256 -> 128
    dropout: float = 0.202      # 调参：0.1 -> 0.202
    lr: float = 1.15e-4         # 调参：1e-4 -> 1.15e-4（再大就容易塌）
    weight_decay: float = 7.49e-6   # 调参：1e-4 -> 7.49e-6
    max_epochs: int = 300   # 只是上限：实际由 EarlyStopping(val_loss, patience=10) 终止
