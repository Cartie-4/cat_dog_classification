from __future__ import annotations

from dataclasses import dataclass, field

from catdog.data.cfg import DEFAULT_IMAGE_SIZE, DataConfig, image_preset


@dataclass
class RNNConfig:
    """RNN（patch embedding + LSTM）的模型配置。

    预处理与其余固定尺寸模型共用 ``image_preset()``（第三轮起训练管线含
    随机旋转 ±15°，在 Normalize 之后），超参与 ``RNN.__init__`` 一一对应。

    关于随机旋转：本模型是"欠拟合型"，旋转让它更差
    -------------------------------------------------
    同一套超参在两种预处理下的验证集表现（500 张，均训练到早停）：

        lr 9.18e-5 / L1 / dropout 0.013 ：无旋转 72.40% -> 有旋转 70.00%
        lr 2.10e-4 / L2 / dropout 0.417 ：       —        -> 有旋转 69.20%

    也就是说旋转对 RNN 是负收益（约 −2.4 pt），而第三轮搜索选出的配置
    在旋转下（69.20%）还不如上一轮配置（70.00%）。因此这里**保留上一轮那组
    超参**（在带旋转的管线下实测更好）。
    如果想完全回到 72.40%，需要给 RNN 单独换一套不含旋转的预处理。
    """

    data: DataConfig = field(
        default_factory=lambda: image_preset(batch_size=64)
    )

    in_channels: int = 3
    num_classes: int = 2
    patch_size: int = 16
    pic_size: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    hidden_dim: int = 256        # patch embedding 输出通道
    bottleneck_dim: int = 128    # patch embedding 的中间通道
    rnn_hidden: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.013
    lr: float = 9.18e-05
    weight_decay: float = 1.05e-04
    max_epochs: int = 300   # 只是上限：实际由 EarlyStopping(val_loss, patience=10) 终止
