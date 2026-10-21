from __future__ import annotations

from typing import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_dim: int,
            bottleneck_dim: int,
            patch_size: int,
            pic_size: Sequence[int],
    ):
        super().__init__()
        assert (pic_size[0] % patch_size == 0 and pic_size[1] % patch_size == 0), f"图片尺寸 {pic_size} 必须能被 patch_size {patch_size} 整除。"
        self.conv1 = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.conv2 = nn.Conv2d(bottleneck_dim, hidden_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RNN(pl.LightningModule):
    """按 patch 嵌入 + RNN 聚合的图片分类模型。

    流程
    ----
    1. 数据管线输出 HWC，``forward`` 里转成 NCHW；
    2. ``Embedding`` 把图片切成 patch，输出 ``(B, hidden_dim, H', W')``；
    3. 展平成 token 序列 ``(B, N, hidden_dim)``；
    4. LSTM/RNN 编码，按时间维做平均池化；
    5. 分类头输出 ``(B, num_classes)``。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        patch_size: int = 16,
        pic_size: Sequence[int] = (192, 192),
        hidden_dim: int = 256,
        bottleneck_dim: int = 128,
        rnn_hidden: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.embedding = Embedding(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            patch_size=patch_size,
            pic_size=pic_size,
        )

        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        feat_dim = rnn_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

        self._init_weights()

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
        # 数据管线是 HWC，卷积需要 NCHW。
        x = x.permute(0, 3, 1, 2).contiguous()

        feat = self.embedding(x)                  # (B, hidden_dim, H', W')
        seq = feat.flatten(2).transpose(1, 2)     # (B, N, hidden_dim)

        out, _ = self.rnn(seq)                    # (B, N, feat_dim)
        pooled = out.mean(dim=1)                  # 时间维平均池化
        return self.head(pooled)

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
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
