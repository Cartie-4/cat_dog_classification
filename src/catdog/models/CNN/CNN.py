"""CNN（PyTorch Lightning 实现）。

- stem: 7x7 conv(stride=2) -> BN -> ReLU -> 3x3 maxpool(stride=2)
- 4 个 stage，每个 stage 由若干个 BasicBlock 组成（默认 (2,2,2,2)）
- 全局平均池化 -> Dropout -> Linear(num_classes)

输入约定：数据管线输出 **HWC** ``(B, H, W, C)``，在 ``forward`` 里转成 NCHW。
对 192x192 输入，特征图尺寸依次为 96 -> 48 -> 24 -> 12 -> 6。
"""

from __future__ import annotations

from typing import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 卷积（padding=1，保持空间尺寸）。"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """ResNet 的基本残差块：conv-bn-relu-conv-bn + shortcut。"""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # 当维度/分辨率变化时，用 1x1 卷积 + BN 调整 shortcut
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


class CNN(pl.LightningModule):
    """二分类 CNN。

    参数
    ----
    in_channels:
        输入通道数（RGB 为 3）。
    num_classes:
        输出类别数（猫狗为 2）。
    layers:
        每个 stage 的 BasicBlock 数量，默认 ``(2, 2, 2, 2)``（类似 ResNet-18）。
    base_width:
        第一个 stage 的通道数，后续 stage 依次翻倍。
    dropout / lr / weight_decay:
        训练相关超参。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        layers: Sequence[int] = (2, 2, 2, 2),
        base_width: int = 64,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay

        if len(layers) != 4:
            raise ValueError(f"layers 需要 4 个 stage，收到 {layers}")

        self.inplanes = base_width

        # stem：192x192 -> 96x96 -> 48x48
        self.conv1 = nn.Conv2d(
            in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个 stage：48 -> 24 -> 12 -> 6
        self.layer1 = self._make_layer(base_width, layers[0])
        self.layer2 = self._make_layer(base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_width * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(base_width * 8, num_classes)

        self._init_weights()

    def _make_layer(
        self, planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """构建一个 stage（若干 BasicBlock）。"""
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
        """卷积用 Kaiming 初始化，BN 用默认的 weight=1 / bias=0。"""
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

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_layer(x)
        return self.fc(x)

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
