import math
import random
from typing import Tuple, Union, Sequence, Iterable


from dataclasses import field, dataclass
import json
import torch
import torch.nn.functional as F

from .types import TransformType

Size = Union[int, Tuple[int, int]]
Range = Union[float, Tuple[float, float]]


def _pair(size: Size) -> Tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    return (int(size[0]), int(size[1]))


def _range(value: Range) -> Tuple[float, float]:
    if isinstance(value, (int, float)):
        return (-float(value), float(value))
    return (float(value[0]), float(value[1]))


def _pad_to(x: torch.Tensor, ch: int, cw: int) -> torch.Tensor:
    h, w = x.shape[0], x.shape[1]
    if ch <= h and cw <= w:
        return x
    pad_h, pad_w = max(0, ch - h), max(0, cw - w)
    return F.pad(x.permute(2, 0, 1), (0, pad_w, 0, pad_h)).permute(1, 2, 0)


def rotate(
    x: torch.Tensor, angle: float, align_corners: bool = False
) -> torch.Tensor:
    if angle == 0:
        return x

    rad = math.radians(angle)
    cos, sin = math.cos(rad), math.sin(rad)
    theta = torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=torch.float32
    )
    theta = theta.unsqueeze(0)

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


class IdentityTransform:
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class Resize:
    def __init__(
        self,
        size: int,
        mode: str = "bilinear",
        antialias: bool = True,
    ):
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

        x_chw = x.permute(2, 0, 1).unsqueeze(0).float()
        out = F.interpolate(
            x_chw, size=(nh, nw), mode=self.mode, antialias=self.antialias
        ).squeeze(0)

        return out.permute(1, 2, 0).to(dtype=x.dtype)


class RandomCrop:
    def __init__(self, size: Size, pad_if_needed: bool = True):
        self.size = _pair(size)
        self.pad_if_needed = pad_if_needed

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ch, cw = self.size
        if self.pad_if_needed:
            x = _pad_to(x, ch, cw)

        h, w = x.shape[0], x.shape[1]
        if ch > h or cw > w:
            raise ValueError(
                f"crop size {(ch, cw)} larger than input {(h, w)}"
            )

        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        return x[top:top + ch, left:left + cw]


class CenterCrop:
    def __init__(self, size: Size, pad_if_needed: bool = True):
        self.size = _pair(size)
        self.pad_if_needed = pad_if_needed

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ch, cw = self.size
        if self.pad_if_needed:
            x = _pad_to(x, ch, cw)

        h, w = x.shape[0], x.shape[1]
        if ch > h or cw > w:
            raise ValueError(
                f"crop size {(ch, cw)} larger than input {(h, w)}"
            )

        top = (h - ch) // 2
        left = (w - cw) // 2
        return x[top:top + ch, left:left + cw]


class RandomFlip:
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
        self.mean = torch.tensor(list(mean),dtype=torch.float32).view(1, 1, -1)
        self.std = torch.tensor(list(std),dtype=torch.float32).view(1, 1, -1).clamp(min=eps)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


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
        return {"name": self.name.value, **self.params}

    @staticmethod
    def from_dict(d: dict) -> "TransformConfig":
        name = TransformType(d["name"])
        params = {k: v for k, v in d.items() if k != "name"}
        return TransformConfig(name=name, params=params)


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
    cfgs: list[TransformConfig],
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
                raise ValueError("Normalize needs mean/std, but stats=None")
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
