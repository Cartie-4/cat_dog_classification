import sys
import math
from typing import Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class DNN(pl.LightningModule):
    def __init__(self, in_dim: int, hidden_dim: int, bottleneck_dim: int, out_dim: int, dropout: float, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.dropout = dropout

        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, bottleneck_dim)
        self.fc3 = torch.nn.Linear(bottleneck_dim,hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # 数据管线输出 HWC 布局 (B, H, W, C)；Linear 需要 (B, features)。
        # 展平放在模型侧（由 pl.Trainer 驱动的 LightningModule 里），
        # 这样 Dataset/DataLoader 保持与布局无关。
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc4(x)
        return x

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer