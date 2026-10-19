import math
import random
from typing import Tuple, Union

import torch
import torch.nn.functional as F

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
