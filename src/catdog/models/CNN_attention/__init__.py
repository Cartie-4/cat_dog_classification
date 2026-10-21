"""CNN + Attention 模型。"""

from .cfg import CNNAttentionConfig
from .CNN_attention import CNNAttention, ResNetTrunk

__all__ = ["CNNAttention", "CNNAttentionConfig", "ResNetTrunk"]
