"""数据预处理中的“几何变换 + 标准化”实现（示例代码）。

设计约定
--------
1. 所有变换的输入 / 输出都是 **HWC 布局** 的 ``torch.Tensor``：
   ``(H, W, C)``，与 ``src/data/types.py`` 里的 ``Pixel1`` 约定保持一致。
   - 图像解码后先转成 ``float`` 张量（取值范围 ``[0, 255]``），再依次经过
     本文件中的变换。
   - 之所以用 HWC，是为了让“读图 -> 几何变换”这一段更直观；真正喂给
     模型时是否转成 CHW / NCHW 由模型侧决定。

2. 本文件同时提供“配置驱动”的工厂：
   ``TransformType`` (枚举) -> ``TransformConfig`` (名称 + 参数)
   -> ``REGISTRY`` (名称 -> 实现类) -> ``build_pipeline`` (配置 -> 可调用对象)。
   这样在 cfg 里只需要写枚举名和少量参数，不需要 import 具体的变换类。

3. ``Normalize`` 的 mean / std 由 ``compute_mean_std`` 在数据集上统计得到，
   **不把数字写进 cfg**。注意这里 **不除以 255**：因为 mean/std 与输入
   都取自 ``[0, 255]`` 同一量纲，``(x - mean) / std`` 已经能得到标准分布。
   （只有在使用 ImageNet 预训练权重、且 mean/std 是 ``[0, 1]`` 量纲的常量时，
   才需要先 ``x / 255``。）
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

# 引入项目里统一的枚举。注意：这里的实现类是示例，生产代码可放回 src/data。
from ..data.types import TransformType

# 尺寸既可以是单个整数（正方形 / 短边），也可以是 (H, W)。
Size = Union[int, Tuple[int, int]]
# 角度既可以是单个数字（对称区间 ±value），也可以是 (lo, hi)。
Range = Union[float, Tuple[float, float]]


# ---------------------------------------------------------------------------
# 小工具函数
# ---------------------------------------------------------------------------
def _pair(size: Size) -> Tuple[int, int]:
    """把 ``int`` 或 ``(h, w)`` 统一成 ``(h, w)`` 元组。"""
    if isinstance(size, int):
        return (size, size)
    return (int(size[0]), int(size[1]))


def _range(value: Range) -> Tuple[float, float]:
    """把 ``float`` 或 ``(lo, hi)`` 统一成 ``(lo, hi)`` 区间。"""
    if isinstance(value, (int, float)):
        return (-float(value), float(value))
    return (float(value[0]), float(value[1]))


def _pad_to(x: torch.Tensor, ch: int, cw: int) -> torch.Tensor:
    """把 HWC 张量在右侧 / 下方补零到至少 ``ch x cw``（够裁剪即可）。"""
    h, w = x.shape[0], x.shape[1]
    if ch <= h and cw <= w:
        return x
    pad_h, pad_w = max(0, ch - h), max(0, cw - w)
    # F.pad 要求 NCHW，这里临时转换再转回 HWC。
    return F.pad(x.permute(2, 0, 1), (0, pad_w, 0, pad_h)).permute(1, 2, 0)


def rotate(x: torch.Tensor, angle: float, align_corners: bool = False) -> torch.Tensor:
    """把 HWC 图像绕中心旋转 ``angle`` 度（双线性插值，边界补零）。"""
    if angle == 0:
        return x

    rad = math.radians(angle)
    cos, sin = math.cos(rad), math.sin(rad)
    # 2x3 仿射矩阵：旋转部分 + 平移部分（这里不平移，绕中心旋转）
    theta = torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=torch.float32
    ).unsqueeze(0)

    x_chw = x.permute(2, 0, 1).unsqueeze(0).float()
    grid = F.affine_grid(theta, x_chw.shape, align_corners=align_corners)
    out = F.grid_sample(
        x_chw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    ).squeeze(0)

    return out.permute(1, 2, 0).to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# 各个变换实现
# ---------------------------------------------------------------------------
class IdentityTransform:
    """不做任何处理，原样返回（用于占位 / 关闭某个增强）。"""

    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Compose:
    """按顺序串联多个变换：``Compose([f1, f2])`` 等价于 ``f2(f1(x))``。"""

    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class Resize:
    """按 **短边** 等比缩放（保持长宽比，长边随之变化）。

    例：``Resize(192)`` 把 ``H x W = 100 x 300`` 缩放成 ``192 x 576``。

    说明：``CNN_attention`` **不使用** 这个变换，它要保留原图尺寸；
    这个类只服务固定尺寸的模型（DNN / CNN / RNN）。
    """

    def __init__(self, size: int, mode: str = "bilinear", antialias: bool = True):
        self.size = int(size)
        self.mode = mode
        self.antialias = antialias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[0], x.shape[1]
        short = min(h, w)
        if short == self.size:
            return x

        scale = self.size / short
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))

        # F.interpolate 要求 NCHW，临时转换再转回 HWC。
        x_chw = x.permute(2, 0, 1).unsqueeze(0).float()
        out = F.interpolate(
            x_chw, size=(nh, nw), mode=self.mode, antialias=self.antialias
        ).squeeze(0)
        return out.permute(1, 2, 0).to(dtype=x.dtype)


class RandomCrop:
    """随机位置裁剪出 ``size`` 大小的区域（训练增强用）。

    ``pad_if_needed=True`` 时，若原图比目标尺寸小会先补零，保证一定能裁。
    """

    def __init__(self, size: Size, pad_if_needed: bool = True):
        self.size = _pair(size)
        self.pad_if_needed = pad_if_needed

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ch, cw = self.size
        if self.pad_if_needed:
            x = _pad_to(x, ch, cw)

        h, w = x.shape[0], x.shape[1]
        if ch > h or cw > w:
            raise ValueError(f"crop size {(ch, cw)} larger than input {(h, w)}")

        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        return x[top:top + ch, left:left + cw]


class CenterCrop:
    """从中心裁剪出 ``size`` 大小的区域（验证 / 推理用）。"""

    def __init__(self, size: Size, pad_if_needed: bool = True):
        self.size = _pair(size)
        self.pad_if_needed = pad_if_needed

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ch, cw = self.size
        if self.pad_if_needed:
            x = _pad_to(x, ch, cw)

        h, w = x.shape[0], x.shape[1]
        if ch > h or cw > w:
            raise ValueError(f"crop size {(ch, cw)} larger than input {(h, w)}")

        top = (h - ch) // 2
        left = (w - cw) // 2
        return x[top:top + ch, left:left + cw]


class RandomFlip:
    """随机翻转。

    在 HWC 布局中：
    - 水平翻转 = 翻转 W 维（dim=1）
    - 垂直翻转 = 翻转 H 维（dim=0）

    默认只做水平翻转（猫狗上下翻转不符合真实分布，故 p_vertical=0）。
    """

    def __init__(self, p_horizontal: float = 0.5, p_vertical: float = 0.0):
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_horizontal:
            x = torch.flip(x, dims=[1])
        if random.random() < self.p_vertical:
            x = torch.flip(x, dims=[0])
        return x


class RandomRotate:
    """随机角度旋转（默认 ±15°）。角度区间的用法见 ``_range``。"""

    def __init__(self, degrees: Range = 15.0, align_corners: bool = False):
        self.degrees = _range(degrees)
        self.align_corners = align_corners

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(*self.degrees)
        return rotate(x, angle, align_corners=self.align_corners)


class Normalize:
    """逐通道标准化：``(x - mean) / std``。

    参数
    ----
    mean, std:
        长度为 ``C`` 的序列（逐通道统计量）。由 ``compute_mean_std`` 在数据集上
        统计得到，量纲与输入一致（这里是 ``[0, 255]``）。

    注意
    ----
    这里 **不再除以 255**。因为 mean/std 本身就是从 ``[0, 255]`` 的像素上统计的，
    ``(x - mean) / std`` 已经是标准分布；若再除 255 反而会破坏量纲一致性。
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float], eps: float = 1e-8):
        # 形状 (1, 1, C)，便于对 HWC 张量做广播。
        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(1, 1, -1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(1, 1, -1).clamp_min(eps)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


# ---------------------------------------------------------------------------
# 配置驱动的工厂：TransformType -> 实现类
# ---------------------------------------------------------------------------
@dataclass
class TransformConfig:
    """一条变换配置。

    - ``name``：用哪个变换（枚举，防止拼写错误）。
    - ``params``：传给实现类构造器的关键字参数，例如
      ``TransformConfig(TransformType.RandomCrop, {"size": 192})``。

    cfg 里只描述“用了什么、参数多少”，不 import 具体类。
    """

    name: TransformType
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为可写入 JSON / 日志的形式（用于复现）。"""
        return {"name": self.name.value, **self.params}

    @staticmethod
    def from_dict(d: dict) -> "TransformConfig":
        """从 ``{"name": "random_crop", "size": 192}`` 反序列化。"""
        d = dict(d)  # 不修改调用方的字典
        name = TransformType(d.pop("name"))
        return TransformConfig(name, d)


# 名称 -> 实现类。新增变换时：加枚举项、写类、在这里注册一行。
REGISTRY: dict[TransformType, type] = {
    TransformType.Identity: IdentityTransform,
    TransformType.Resize: Resize,
    TransformType.RandomCrop: RandomCrop,
    TransformType.CenterCrop: CenterCrop,
    TransformType.RandomFlip: RandomFlip,
    TransformType.RandomRotate: RandomRotate,
    TransformType.Normalize: Normalize,
}


def build_transform(cfg: TransformConfig):
    """根据一条配置实例化一个变换。未知名称或参数错误会立即抛异常。"""
    cls = REGISTRY[cfg.name]          # 未知枚举 -> KeyError
    return cls(**cfg.params)          # 参数名/类型错误 -> TypeError


def build_pipeline(
    cfgs: Sequence[TransformConfig],
    stats: tuple[Sequence[float], Sequence[float]] | None = None,
) -> Compose:
    """把配置列表构建成 ``Compose``。

    ``Normalize`` 比较特殊：它需要 mean/std，而 cfg 里不写具体数字，
    所以在构建时通过 ``stats=(mean, std)`` 注入。
    """
    pipeline = []
    for cfg in cfgs:
        if cfg.name is TransformType.Normalize:
            if stats is None:
                raise ValueError("构建含 Normalize 的管线时必须提供 stats=(mean, std)")
            mean, std = stats
            pipeline.append(Normalize(mean, std))
        else:
            pipeline.append(build_transform(cfg))
    return Compose(pipeline)


def pipeline_signature(cfgs: Sequence[TransformConfig]) -> str:
    """生成管线的“几何部分”签名，用作统计量缓存键。

    统计量不依赖随机增强，也不依赖 Normalize 本身，因此这里剔除 Normalize；
    ``setup()`` 会用 **验证管线**（确定性）调用它。
    """
    payload = [
        c.to_dict() for c in cfgs if c.name is not TransformType.Normalize
    ]
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 数据集统计量：逐通道 mean / std
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_mean_std(
    tensors: Iterable[torch.Tensor], eps: float = 1e-8
) -> tuple[list[float], list[float]]:
    """在一批 HWC 张量上统计逐通道均值与标准差。

    参数
    ----
    tensors:
        可迭代对象，每次产出一个 ``(H, W, C)`` 的 ``[0, 255]`` 浮点张量
        （通常是“去掉 Normalize 之后”的管线输出）。

    返回
    ----
    ``(mean, std)``，各为长度 ``C`` 的 Python list。

    实现上用“和 / 平方和 / 像素数”一次性累积，避免把整份数据读进内存。
    使用 float64 累积以减小数值误差。
    """
    total = None
    total_sq = None
    count = 0

    for x in tensors:
        x = x.to(torch.float64)
        c = x.shape[-1]
        if total is None:
            total = torch.zeros(c, dtype=torch.float64)
            total_sq = torch.zeros(c, dtype=torch.float64)

        total += x.sum(dim=(0, 1))
        total_sq += (x * x).sum(dim=(0, 1))
        count += x.shape[0] * x.shape[1]

    if count == 0 or total is None:
        raise ValueError("没有可用于统计的数据")

    mean = total / count
    var = total_sq / count - mean * mean
    std = var.clamp_min(eps).sqrt()
    return mean.tolist(), std.tolist()
