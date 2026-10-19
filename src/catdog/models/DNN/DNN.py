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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer