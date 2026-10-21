"""CNN + 单头注意力聚合（PyTorch Lightning 实现）。

整体流程
--------
1. 数据管线输出变尺寸 HWC 图片 + padding mask；
2. ``ResNetTrunk``（复用 CNN 的 ``BasicBlock``）输出特征图 ``(B, d, H', W')``；
3. 展平成 token 序列 ``X = (B, N, d)``；
4. 可学习投影得到 ``K = X @ W_K``、``V = X @ W_V``，用全局可学习向量 ``Q``
   与 ``K`` 计算注意力权重；
5. 用 mask 屏蔽 padding token 后 softmax，聚合得到每张图一个向量；
6. FNN 输出分类 logits。
"""

from __future__ import annotations

from typing import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 CNN 的残差块，避免重复实现。
from catdog.models.CNN.CNN import BasicBlock


class ResNetTrunk(nn.Module):
    """ResNet 风格的骨干网络（只输出特征图，不含分类头）。

    输入 NCHW，输出 ``(B, base_width*8, H', W')``，总下采样 /32。
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_width: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
    ):
        super().__init__()
        if len(layers) != 4:
            raise ValueError(f"layers 需要 4 个 stage，收到 {layers}")

        self.inplanes = base_width
        self.out_channels = base_width * 8

        # stem：/4
        self.conv1 = nn.Conv2d(
            in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个 stage：/8 /16 /32 /32（layer1 在 /4 基础上不再下采样）
        self.layer1 = self._make_layer(base_width, layers[0])
        self.layer2 = self._make_layer(base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_width * 8, layers[3], stride=2)

        self._init_weights()

    def _make_layer(
        self, planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class CNNAttention(pl.LightningModule):
    """CNN + 单头注意力聚合的图片分类模型。

    注意力采用单 query 形式：

    - ``W_K``、``W_V``：可学习投影矩阵 ``(d, d)``，全局共享；
    - ``Q``：可学习向量 ``(d,)``，全局共享（不是 per-head、不是 per-image）；
    - ``scores = K·Q / sqrt(d)``，mask 掉 padding 后 softmax；
    - ``image_vec = sum_i alpha_i * V_i``；
    - FNN 输出 ``num_classes``。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_width: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
        fnn_hidden: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.trunk = ResNetTrunk(
            in_channels=in_channels,
            base_width=base_width,
            layers=layers,
        )
        d = self.trunk.out_channels

        # 单头注意力的可学习参数（全局共享）
        self.w_k = nn.Parameter(torch.empty(d, d))
        self.w_v = nn.Parameter(torch.empty(d, d))
        self.q = nn.Parameter(torch.empty(d))

        self.norm = nn.LayerNorm(d)
        self.fnn = nn.Sequential(
            nn.Linear(d, fnn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fnn_hidden, num_classes),
        )

        self._init_attention()

    def _init_attention(self) -> None:
        nn.init.trunc_normal_(self.w_k, std=0.02)
        nn.init.trunc_normal_(self.w_v, std=0.02)
        nn.init.trunc_normal_(self.q, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # HWC -> NCHW
        x = x.permute(0, 3, 1, 2).contiguous()

        feat = self.trunk(x)                      # (B, d, H', W')
        b, d, hf, wf = feat.shape

        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, d)

        # 把像素级 mask 对齐到 token 级：只要覆盖到真实像素就算有效。
        token_mask = F.adaptive_max_pool2d(
            mask.float().unsqueeze(1), (hf, wf)
        ).squeeze(1)
        token_mask = token_mask.flatten(1) > 0    # (B, N) bool

        k = tokens @ self.w_k                     # (B, N, d)
        v = tokens @ self.w_v                     # (B, N, d)

        # scores = <k, q> / sqrt(d)
        scores = torch.einsum("bnd,d->bn", k, self.q) / (d ** 0.5)
        scores = scores.masked_fill(~token_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1)      # (B, N)

        image_vec = torch.einsum("bn,bnd->bd", alpha, v)  # (B, d)
        image_vec = self.norm(image_vec)
        return self.fnn(image_vec)

    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self(x, mask)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self(x, mask)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
