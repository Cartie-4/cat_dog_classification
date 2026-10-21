"""CNN 模型（ResNet-18 风格）。"""

from .cfg import CNNConfig
from .CNN import CNN

__all__ = ["CNN", "CNNConfig"]
