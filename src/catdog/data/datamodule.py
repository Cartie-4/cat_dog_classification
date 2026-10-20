from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .types import DATASET_DIR, TransformType
from .cfg import DataConfig, MIN_SHORT_SIDE, image_preset 
from .dataset import CatDogDataset, make_collate_fn
from .transforms import build_pipeline, compute_mean_std, pipeline_signature

DEFAULT_STATS_PATH = Path(__file__).parent / "normalize_stats.json"

class CatDogDataModule(pl.LightningDataModule):    
    """猫狗分类的 LightningDataModule。

    参数
    ----
    data_cfg:
        数据配置。默认使用固定尺寸的 ``image_preset()``；
        若想用 ``CNN_attention`` 的预处理，传入
        ``CNNAttentionConfig().data``（见 ``models/CNN_attention/cfg.py``）。
    root:
        数据根目录。默认取 ``DATASET_DIR["kaggle"]``（可用 ``CATDOG_DATA_DIR`` 覆盖）。
    num_workers:
        DataLoader 工作进程数。Windows 下建议保持 0。
    stats_path:
        统计量缓存文件路径。
    stats_max_samples:
        计算统计量时最多扫描多少张训练图（None 表示全部）。
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

    def prepare_data(self) -> None:
        """下载 / 预处理数据。本项目的图片已在本地，无需额外操作。"""
        return None 

    def setup(self, stage: Optional[str] = None) -> None:
        """构建数据集，只构建一次，train/val阶段共用"""
        if self._train_ds is not None and self._val_ds is not None:
            return

        self.stats = self._resolve_stats()

        train_tf = build_pipeline(self.data_cfg.train, stats=self.stats)
        val_tf = build_pipeline(self.data_cfg.val, stats=self.stats)

        self._train_ds = CatDogDataset(
            "train", root=self.root, transform=train_tf
        )    
        self._val_ds = CatDogDataset(
            "val", root=self.root, transform=val_tf
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """释放数据集。"""
        self._train_ds = None
        self._val_ds = None

    def _make_loader(
            self,
            dataset: CatDogDataset,
            train: bool
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=train,
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

    def _cache_key(self) -> str:
        signature = pipeline_signature(self.data_cfg.val)
        # 键里带上 root 与 min_short_side，避免换数据集/阈值后误用旧统计量。
        raw = f"train|{self.root}|{MIN_SHORT_SIDE}|{signature}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    def _load_cache(self) -> dict:
        if not self.stats_path.exists():
            return {}
        with open(self.stats_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_cache(self, cache: dict) -> None:
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    def _resolve_stats(self) -> tuple[list[float], list[float]]:
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