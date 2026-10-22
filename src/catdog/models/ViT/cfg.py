from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DEFAULT_IMAGE_SIZE, DataConfig, image_preset


@dataclass
class ViTConfig:
    """ViT 的模型配置。

    预处理与 DNN / CNN / RNN 共用 ``image_preset()``（短边缩放到 192 -> 随机裁剪 192），
    超参与 ``ViT.__init__`` 的参数一一对应。

    注意 ``image_size`` 必须和 ``data`` 里的裁剪边长一致：位置编码的长度是
    ``(image_size // patch_size) ** 2 + 1``，两者对不上会在 forward 里报形状错误。

    调参结论（每组超参训练到**自己的早停点**再比较）
    ----------------------------------------------
    - 最优：lr 1.9e-4、embed_dim 128、depth 4、patch 16、dropout 0.083、
      weight_decay 1.24e-4、batch 32，**73.60%**（约 37 轮到早停）；
    - 同一配置重复跑一次是 72.20%，即本环境的单次运行噪声约 ±1.4 pt；
    - 这个模型在小数据集上明显吃亏：从零训练 2000 张图，最好的也才 73.6%，
      远低于 CNN 的 87.6%。Transformer 缺卷积的局部性/平移等变先验，需要更多数据或预训练。
    - patch_size=32（36 个 token）最快但更差（71.00%，且要 120 轮）。
    """

    data: DataConfig = field(default_factory=image_preset)

    in_channels: int = 3
    num_classes: int = 2
    image_size: int = DEFAULT_IMAGE_SIZE
    patch_size: int = 16
    embed_dim: int = 128            # 调参：192 -> 128
    depth: int = 4                  # 调参：6 -> 4
    num_heads: int = 4              # 必须整除 embed_dim；128/4=32
    mlp_ratio: float = 4.0
    dropout: float = 0.083          # 调参：0.1 -> 0.083
    attn_dropout: float = 0.0
    lr: float = 1.9e-4              # 调参：3e-4 -> 1.9e-4（这是 warmup 后的峰值 lr）
    weight_decay: float = 1.24e-4   # 调参：5e-2 -> 1.24e-4
    warmup_frac: float = 0.1        # 前 10% 的 step 线性 warmup（0 -> lr）
    min_lr_ratio: float = 0.0       # cosine 衰减到 0
    max_epochs: int = 300   # 只是上限：实际由 EarlyStopping(val_loss, patience=10) 终止
