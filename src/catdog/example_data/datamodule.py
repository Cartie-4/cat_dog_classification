"""用 PyTorch Lightning 的 ``LightningDataModule`` 组织示例预处理。

这个 DataModule 负责：
1. 根据 ``DataConfig`` 构建训练 / 验证数据集；
2. 计算（并缓存）逐通道 mean/std，供 ``Normalize`` 使用；
3. 按配置返回固定尺寸或“变尺寸 + mask”的 DataLoader。

关于统计量
----------
- 统计只在 **训练集** 上做；
- 使用 **验证管线**（确定性，不含随机增强）去掉 ``Normalize`` 后扫描，保证
  每次计算结果稳定；
- 结果缓存到 ``normalize_stats.json``，按“管线签名”区分：
  固定尺寸模型与 ``CNN_attention`` 各有一份。
- 缓存文件放在本目录下，可随代码提交以保证复现。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..data.types import DATASET_DIR, TransformType

from .config import DataConfig, image_preset
from .dataset import CatDogDataset, make_collate_fn
from .transforms import build_pipeline, compute_mean_std, pipeline_signature

# 默认的统计量缓存路径：与代码同目录，便于复现。
DEFAULT_STATS_PATH = Path(__file__).parent / "normalize_stats.json"


class CatDogDataModule(pl.LightningDataModule):
    """猫狗分类的 LightningDataModule。

    参数
    ----
    data_cfg:
        数据配置。默认使用固定尺寸的 ``image_preset()``；
        若想用 ``CNN_attention`` 的预处理，传入 ``attention_preset()``。
    root:
        数据根目录，默认 ``data/``。
    num_workers:
        DataLoader 工作进程数。Windows 下建议保持 0。
    stats_path:
        统计量缓存文件路径。
    stats_max_samples:
        计算统计量时最多扫描多少张训练图（None 表示全部）。
        示例里可设小一点以加快首次运行。
    """

    def __init__(
        self,
        data_cfg: Optional[DataConfig] = None,
        root: str = DATASET_DIR["kaggle"],
        num_workers: int = 0,
        stats_path: Path = DEFAULT_STATS_PATH,
        stats_max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_cfg = data_cfg if data_cfg is not None else image_preset()
        self.root = root
        self.num_workers = num_workers
        self.stats_path = Path(stats_path)
        self.stats_max_samples = stats_max_samples

        self._train_ds: Optional[CatDogDataset] = None
        self._val_ds: Optional[CatDogDataset] = None
        self.stats: Optional[tuple[list[float], list[float]]] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        """下载 / 预处理数据。本项目的图片已在本地，无需额外操作。"""
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        """构建数据集（只构建一次，fit / validate 阶段共用）。"""
        if self._train_ds is not None and self._val_ds is not None:
            return

        # 先拿到标准化统计量，再注入到训练 / 验证管线里的 Normalize。
        self.stats = self._resolve_stats()

        train_tf = build_pipeline(self.data_cfg.train, self.stats)
        val_tf = build_pipeline(self.data_cfg.val, self.stats)

        self._train_ds = CatDogDataset(
            "train", root=self.root, transform=train_tf
        )
        self._val_ds = CatDogDataset(
            "val", root=self.root, transform=val_tf
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning 生命周期钩子：释放引用。"""
        self._train_ds = None
        self._val_ds = None

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    def _make_loader(self, dataset: CatDogDataset, train: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=train,                       # 只有训练集打乱
            num_workers=self.num_workers,
            collate_fn=make_collate_fn(self.data_cfg.variable_size),
            pin_memory=False,
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            self.setup("fit")
        return self._make_loader(self._train_ds, train=True)

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            self.setup("validate")
        return self._make_loader(self._val_ds, train=False)

    # ------------------------------------------------------------------
    # 统计量：计算 / 缓存
    # ------------------------------------------------------------------
    def _cache_key(self) -> str:
        """用“训练集 + 验证管线（去掉 Normalize）”生成缓存键。"""
        signature = pipeline_signature(self.data_cfg.val)
        raw = f"train|{signature}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    def _load_cache(self) -> dict:
        if not self.stats_path.exists():
            return {}
        with open(self.stats_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_cache(self, cache: dict) -> None:
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    def _resolve_stats(self) -> tuple[list[float], list[float]]:
        """优先读缓存；没有则计算一次并写回。"""
        key = self._cache_key()
        cache = self._load_cache()

        if key in cache:
            entry = cache[key]
            return entry["mean"], entry["std"]

        mean, std = self._compute_stats()
        cache[key] = {"mean": mean, "std": std}
        self._save_cache(cache)
        return mean, std

    def _compute_stats(self) -> tuple[list[float], list[float]]:
        """在训练集上统计逐通道 mean/std（不含 Normalize，确定性管线）。"""
        # 用验证管线的几何部分：Resize + CenterCrop 是确定性的，适合统计。
        # 对 CNN_attention 来说验证管线只有 Normalize，去掉后为空 -> 保持原图。
        stats_cfgs = [
            c for c in self.data_cfg.val if c.name is not TransformType.Normalize
        ]
        stats_tf = build_pipeline(stats_cfgs)

        dataset = CatDogDataset("train", root=self.root, transform=stats_tf)
        n = len(dataset)
        if self.stats_max_samples is not None:
            n = min(n, self.stats_max_samples)

        # 只需要图像张量，不需要 label。
        tensors = (dataset[i][0] for i in range(n))
        return compute_mean_std(tensors)
