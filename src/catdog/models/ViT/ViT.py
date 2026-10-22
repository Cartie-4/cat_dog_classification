"""ViT（Vision Transformer，PyTorch Lightning 实现）。

整体流程
--------
1. 数据管线输出 **HWC** ``(B, H, W, C)``，在 ``forward`` 里转成 NCHW；
2. ``PatchEmbedding`` 用 stride = patch_size 的卷积把图切成不重叠的 patch，
   并线性映射到 ``embed_dim``，得到 ``(B, N, D)``；
3. 前面拼一个可学习的 ``[CLS]`` token，再加可学习的位置编码；
4. 过 ``depth`` 个 Pre-LN Transformer block（多头自注意力 + MLP）；
5. 取 ``[CLS]`` 位置的输出，LayerNorm 后接线性分类头。

对 192x192 输入、patch_size=16：``N = 12x12 = 144`` 个 patch，加 CLS 共 145 个 token。

和 ``CNN_attention`` 的区别
--------------------------
``CNN_attention`` 是「卷积主干 + 单个可学习 query 做加权池化」，聚合是线性的；
这里是完整的 Transformer：**多头自注意力 + 池化前的 token 级 MLP**，并且有显式位置编码。
所以它的容量和表达能力都更强，代价是更像"没有卷积先验"，更依赖数据量和正则。
"""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """把图片切成 patch 并映射到 ``embed_dim`` 维。

    用 ``kernel_size = stride = patch_size`` 的卷积实现，等价于「切块 + 线性投影」，
    但比自己 reshape 再 matmul 更快。
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        image_size: int,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} 必须能被 patch_size={patch_size} 整除"
            )
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> ``(B, N, D)``。"""
        x = self.proj(x)                      # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)      # (B, N, D)
        return x


class MLP(nn.Module):
    """Transformer 里的 token 级前馈网络：Linear -> GELU -> Linear。"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN 结构：``x = x + Attn(LN(x))``，``x = x + MLP(LN(x))``。

    Pre-LN（先归一化再进子层）比原始论文的 Post-LN 更好训，是小数据集上的常用做法。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(pl.LightningModule):
    """用于猫狗二分类的 Vision Transformer。

    参数
    ----
    in_channels:
        输入通道数（RGB 为 3）。
    num_classes:
        输出类别数（猫狗为 2）。
    image_size / patch_size:
        输入边长与切块大小，两者必须整除；``(image_size // patch_size) ** 2`` 是 token 数。
        **必须与数据管线里 ``image_preset(size)`` 的 size 一致**，否则位置编码形状对不上。
    embed_dim / depth / num_heads:
        token 维度、Transformer block 层数、注意力头数。
    mlp_ratio:
        MLP 隐层 = ``embed_dim * mlp_ratio``。
    dropout / attn_dropout:
        通用 dropout 与注意力内部 dropout。
    lr / weight_decay:
        训练超参。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        image_size: int = 192,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        lr: float = 3e-4,
        weight_decay: float = 5e-2,
        warmup_frac: float = 0.1,
        min_lr_ratio: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} 必须能被 num_heads={num_heads} 整除"
            )
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.warmup_frac = warmup_frac
        self.min_lr_ratio = min_lr_ratio

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            image_size=image_size,
        )
        num_tokens = self.patch_embed.num_patches + 1  # +1 是 CLS

        # [CLS] token 与位置编码都用可学习参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """ViT 的标准初始化：卷积/线性用截断正态，LayerNorm 权重为 1、偏置为 0。"""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 数据管线是 HWC，卷积/patch 投影需要 NCHW。
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.patch_embed(x)                            # (B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)     # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                     # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])                          # 取 CLS 位置

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Adam + **线性 warmup + cosine 衰减**（按 step 更新）。

        为什么 ViT 需要这个：
        - 没有卷积的局部性/平移先验，训练早期梯度尺度很不稳定，
          直接用常数 lr 会让注意力 logits 迅速饱和（这也是调参时大 lr 直接塌缩的原因）；
        - warmup 让模型先在小 lr 下把 patch embedding / 位置编码 / 注意力模式"扶上路"；
        - cosine 衰减让后期自动收小步长，避免在最优点附近来回震荡。

        调度总步数取 ``trainer.estimated_stepping_batches``（= epoch 数 × 每轮 batch 数），
        所以 **max_epochs 就是 cosine 的衰减地平线**：改 epoch 数会一起改曲线形状。
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = int(total_steps * self.warmup_frac)
        min_ratio = self.min_lr_ratio
        base_lr = self.lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # 线性 warmup：0 -> base_lr
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
