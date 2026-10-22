from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DataConfig, image_preset


@dataclass
class CNNConfig:
    """CNN的模型配置。

    预处理与其余固定尺寸模型共用 ``image_preset()``；超参与 ``CNN.__init__``
    的参数一一对应。

    调参结论（第二轮：每组超参训练到**自己的早停点**再比较）
    -------------------------------------------------------
    这一轮的结论和第一轮（固定 8 epoch）**完全相反**，原因是训练预算：

    - 第一轮 8 epoch 时：lr 3.22e-4 + width 32 最好（77.15%），lr >= 7e-3 只有 67~68%，
      于是误判成"大 lr 不行"；
    - 本轮给足预算（早停，约 90 轮）：**lr 8.3e-3 + width 96 + batch 64** 达到 **87.60%**，
      lr 2.96e-3 得 86.20%，而第一轮的冠军 lr 3.22e-4 只有 81.80%。
    - 结论：lr 与训练轮数强耦合。大 lr 需要更多轮才能收敛，但收敛后精度更高；
      之前"小 lr 更好"的结论是短预算造成的假象。
    """

    data: DataConfig = field(
        default_factory=lambda: image_preset(batch_size=64)
    )

    in_channels: int = 3
    num_classes: int = 2
    layers: tuple[int, int, int, int] = (2, 2, 2, 2)  # 每个 stage 的 BasicBlock 数
    base_width: int = 96        # 调参：32 -> 96（给足轮数时更宽更好）
    dropout: float = 0.178      # 调参：0.405 -> 0.178
    lr: float = 8.3e-3          # 调参：3.22e-4 -> 8.3e-3（需要更多轮，约 91 轮到早停）
    weight_decay: float = 4.47e-6   # 调参：2.28e-4 -> 4.47e-6
    max_epochs: int = 300   # 只是上限：实际由 EarlyStopping(val_loss, patience=10) 终止
