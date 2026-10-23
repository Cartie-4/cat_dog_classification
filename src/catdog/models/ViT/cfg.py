from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DEFAULT_IMAGE_SIZE, DataConfig, image_preset


@dataclass
class ViTConfig:
    """ViT 的模型配置。

    预处理与 DNN / CNN / RNN 共用 ``image_preset()``（第三轮起训练管线含
    随机旋转 ±15°，在 Normalize 之后），超参与 ``ViT.__init__`` 一一对应。

    ``image_size`` 必须与 ``data`` 的裁剪边长一致：位置编码长度是
    ``(image_size // patch_size) ** 2 + 1``。

# 第三轮调参（image_preset 增加 RandomRotate ±15° 之后重新搜索）
# 最优 trial：val_acc=74.00%，收敛 67 轮
    ``configure_optimizers`` 里带线性 warmup + cosine 衰减（按 step），
    ``warmup_frac`` 是 warmup 占全部 step 的比例。
    """

    data: DataConfig = field(
        default_factory=lambda: image_preset(batch_size=32)
    )

    in_channels: int = 3
    num_classes: int = 2
    image_size: int = DEFAULT_IMAGE_SIZE
    patch_size: int = 16
    embed_dim: int = 128
    depth: int = 8
    num_heads: int = 4              # 必须整除 embed_dim
    mlp_ratio: float = 4.0
    dropout: float = 0.028
    attn_dropout: float = 0.0
    lr: float = 0.000152
    weight_decay: float = 0.000131
    warmup_frac: float = 0.1
    min_lr_ratio: float = 0.0
    max_epochs: int = 300   # 只是上限：实际由 EarlyStopping(val_loss, patience=10) 终止
