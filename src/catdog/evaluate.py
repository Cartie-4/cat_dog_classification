"""从 checkpoint 加载模型，输出验证集上的分类结果与混淆矩阵。

用法::

    # 直接指定 ckpt
    python -m catdog.evaluate --ckpt outputs/2026-10-21_12-00-00/checkpoints/CNN-epoch=07-val_loss=0.2134.ckpt

    # 指定 checkpoints 目录（自动挑 val_loss 最低的那个）
    python -m catdog.evaluate --ckpt outputs/2026-10-21_12-00-00/checkpoints --split val

    # 训练集也看一遍（用确定性的验证管线，不带随机增强）
    python -m catdog.evaluate --ckpt <...> --split both

    # 显式指定模型类型 / 图片尺寸（正常情况下都能自动推断）
    python -m catdog.evaluate --ckpt <...> --model cnn --image-size 192

输出（默认写到 ``outputs/eval/<ckpt名>_<split>/``）：

- 终端打印混淆矩阵、逐类指标、错分最多的样本；
- ``confusion_matrix.png``：计数 + 按真实类归一化的两张热力图（用 Pillow 绘制）；
- ``confusion_matrix.txt``：纯文本混淆矩阵（与终端输出一致，便于粘贴）；
- ``metrics.json``：accuracy / precision / recall / F1 / AUC 等；
- ``predictions.csv``：逐样本的真实标签、预测、置信度。

实现要点
--------
- 模型结构直接从 ckpt 里的 ``hyper_parameters`` 反推，**不需要手动对齐超参**；
  模型类型（dnn / cnn / rnn / cnn_attention）也能自动识别。
- 预处理用该模型族 cfg 里的**验证管线**（确定性），Normalize 的 mean/std 由
  ``CatDogDataModule`` 按缓存键解析，与训练时保持一致。
- 画图只用 Pillow，不依赖 matplotlib（Pillow 是项目已有依赖，且更轻量）。
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from catdog.data import (
    CLASSES,
    DEFAULT_IMAGE_SIZE,
    CatDogDataModule,
    CatDogDataset,
    build_pipeline,
    image_preset,
    make_collate_fn,
)
from catdog.models.CNN import CNN
from catdog.models.CNN_attention import CNNAttention
from catdog.models.CNN_attention.cfg import _attention_data
from catdog.models.DNN import DNN
from catdog.models.RNN import RNN
from catdog.models.ViT import ViT

MODEL_CLASSES = {
    "dnn": DNN,
    "cnn": CNN,
    "rnn": RNN,
    "cnn_attention": CNNAttention,
    "vit": ViT,
}


# ---------------------------------------------------------------------------
# checkpoint / 模型
# ---------------------------------------------------------------------------
def resolve_ckpt(path: Path) -> Path:
    """``path`` 可以是 ckpt 文件，也可以是 checkpoints 目录。"""
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"找不到 checkpoint: {path}")

    ckpts = sorted(path.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"目录里没有 .ckpt 文件: {path}")

    # 文件名里带 val_loss= 的，取最小的那个（ModelCheckpoint 的 save_top_k=1）
    def loss_of(p: Path) -> float:
        stem = p.stem
        if "val_loss=" in stem:
            try:
                return float(stem.split("val_loss=")[-1])
            except ValueError:
                pass
        return float("inf")

    ranked = sorted(ckpts, key=loss_of)
    return ranked[0]


def load_ckpt(path: Path) -> dict[str, Any]:
    """加载 ckpt。

    torch>=2.6 的 ``weights_only`` 默认为 True，而 Lightning ckpt 里只有
    张量和基础类型，正常可以直接读；老格式读取失败时回退到完整反序列化。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # pragma: no cover - 兼容旧 ckpt
        return torch.load(path, map_location="cpu", weights_only=False)


def infer_model_name(hparams: dict[str, Any]) -> str:
    """从超参键名反推模型类型。"""
    keys = set(hparams)
    # 注意顺序：ViT 也有 patch_size，必须先于 rnn 判断
    if "embed_dim" in keys and "depth" in keys:
        return "vit"
    if "fnn_hidden" in keys:
        return "cnn_attention"
    if "rnn_hidden" in keys or "patch_size" in keys:
        return "rnn"
    if "in_dim" in keys:
        return "dnn"
    if "layers" in keys and "base_width" in keys:
        return "cnn"
    raise ValueError(f"无法从超参推断模型类型，请用 --model 指定。超参键: {sorted(keys)}")


def build_model(model_name: str, hparams: dict[str, Any]) -> pl.LightningModule:
    """按 ckpt 里的超参重建模型结构（不加载权重）。"""
    cls = MODEL_CLASSES[model_name]
    sig = inspect.signature(cls.__init__)
    kwargs = {
        key: value
        for key, value in hparams.items()
        if key in sig.parameters and key != "self"
    }
    return cls(**kwargs)


def _dnn_size_from_in_dim(hparams: dict[str, Any]) -> Optional[int]:
    """DNN 的输入维度是 H*W*3，反推出正方形边长。"""
    in_dim = hparams.get("in_dim")
    if not in_dim:
        return None
    side = int(round((in_dim / 3) ** 0.5))
    return side if side * side * 3 == int(in_dim) else None


def build_data_cfg(
    model_name: str,
    hparams: dict[str, Any],
    batch_size: int,
    image_size: Optional[int] = None,
):
    """构造评估用的数据配置（只用其中的 ``val`` 管线）。"""
    if model_name == "cnn_attention":
        return _attention_data(batch_size)

    size = image_size
    if size is None:
        if model_name == "dnn":
            size = _dnn_size_from_in_dim(hparams)
        elif model_name == "rnn":
            pic = hparams.get("pic_size")
            if pic:
                size = int(pic[0])
        elif model_name == "vit":
            size = hparams.get("image_size")
        if size is None:
            size = DEFAULT_IMAGE_SIZE
    return image_preset(int(size), batch_size)


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(
    model: pl.LightningModule,
    loader: DataLoader,
    device: torch.device,
    use_mask: bool,
    limit_batches: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(probs, labels)``，``probs`` 形状 ``(N, num_classes)``。"""
    model.eval()
    model.to(device)

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for step, batch in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break
        if use_mask:
            x, mask, y = batch
            logits = model(x.to(device), mask.to(device))
        else:
            x, y = batch
            logits = model(x.to(device))

        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    if not all_probs:
        raise RuntimeError("没有取到任何样本，检查 --limit-batches / 数据路径")

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def classification_metrics(cm: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """由混淆矩阵算 accuracy / 逐类 precision-recall-F1 / macro 平均。"""
    num_classes = cm.shape[0]
    total = int(cm.sum())
    correct = int(np.trace(cm))

    per_class = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        support = int(cm[c, :].sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        per_class.append(
            {
                "class": names[c],
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

    macro = {
        key: round(float(np.mean([m[key] for m in per_class])), 4)
        for key in ("precision", "recall", "f1")
    }
    return {
        "num_samples": total,
        "accuracy": round(_safe_div(correct, total), 4),
        "per_class": per_class,
        "macro_avg": macro,
    }


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """正类（下标 1）的 ROC-AUC，用 Mann-Whitney U 秩和公式实现，正确处理并列。"""
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # 并列取平均秩
        i = j + 1

    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(float(auc), 4)


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def print_confusion(cm: np.ndarray, names: Sequence[str]) -> None:
    print("\n" + confusion_text(cm, names))


def print_metrics(metrics: dict[str, Any]) -> None:
    print(f"\n总体准确率: {metrics['accuracy'] * 100:.2f}%  ({metrics['num_samples']} 张)")
    if "roc_auc" in metrics and not np.isnan(metrics["roc_auc"]):
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\n{'class':<10}{'support':>9}{'precision':>11}{'recall':>9}{'f1':>9}")
    for m in metrics["per_class"]:
        print(
            f"{m['class']:<10}{m['support']:>9}{m['precision']:>11.4f}"
            f"{m['recall']:>9.4f}{m['f1']:>9.4f}"
        )
    macro = metrics["macro_avg"]
    print(
        f"{'macro':<10}{'':>9}{macro['precision']:>11.4f}"
        f"{macro['recall']:>9.4f}{macro['f1']:>9.4f}"
    )


_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """找一个可用的 TrueType 字体，找不到就用 Pillow 内置位图字体。"""
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)  # Pillow >= 10.1
    except TypeError:  # pragma: no cover - 老版本 Pillow
        return ImageFont.load_default()


def _cell_color(t: float) -> tuple[int, int, int]:
    """``t`` 从 0（白）到 1（深蓝）线性插值，模仿 matplotlib 的 Blues。"""
    t = min(max(t, 0.0), 1.0)
    light = (247, 251, 255)
    dark = (8, 48, 107)
    return tuple(int(light[i] + (dark[i] - light[i]) * t) for i in range(3))


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    matrix: np.ndarray,
    names: Sequence[str],
    origin: tuple[int, int],
    cell: int,
    subtitle: str,
    fmt: str,
    scale: Optional[float],
) -> None:
    """画一个混淆矩阵面板：``origin`` 是矩阵左上角像素坐标。"""
    font_cell = _load_font(int(cell * 0.32))
    font_label = _load_font(int(cell * 0.30))
    font_title = _load_font(int(cell * 0.34))
    ox, oy = origin
    rows, cols = matrix.shape

    peak = float(matrix.max()) if scale is None else 1.0
    # 面板标题（在矩阵上方居中）
    title_box = draw.textbbox((0, 0), subtitle, font=font_title)
    draw.text(
        (ox + (cols * cell - (title_box[2] - title_box[0])) / 2, oy - int(cell * 0.95)),
        subtitle,
        fill=(30, 30, 30),
        font=font_title,
    )

    for j, name in enumerate(names):  # 列标签只写类名，避免相邻标签互相压到
        box = draw.textbbox((0, 0), name, font=font_label)
        draw.text(
            (ox + j * cell + (cell - (box[2] - box[0])) / 2, oy - int(cell * 0.45)),
            name,
            fill=(60, 60, 60),
            font=font_label,
        )

    for i, name in enumerate(names):  # 行标签
        text = f"true: {name}"
        box = draw.textbbox((0, 0), text, font=font_label)
        draw.text(
            (ox - (box[2] - box[0]) - int(cell * 0.15),
             oy + i * cell + (cell - (box[3] - box[1])) / 2),
            text,
            fill=(60, 60, 60),
            font=font_label,
        )

    for i in range(rows):
        for j in range(cols):
            value = float(matrix[i, j])
            t = value / peak if peak > 0 else 0.0
            x0, y0 = ox + j * cell, oy + i * cell
            draw.rectangle(
                [x0, y0, x0 + cell, y0 + cell],
                fill=_cell_color(t),
                outline=(255, 255, 255),
                width=2,
            )
            label = f"{value:.1%}" if fmt == "pct" else f"{int(value)}"
            box = draw.textbbox((0, 0), label, font=font_cell)
            draw.text(
                (
                    x0 + (cell - (box[2] - box[0])) / 2,
                    y0 + (cell - (box[3] - box[1])) / 2 - box[1],
                ),
                label,
                fill=(255, 255, 255) if t > 0.55 else (16, 16, 16),
                font=font_cell,
            )

    # 面板外框
    draw.rectangle(
        [ox, oy, ox + cols * cell, oy + rows * cell],
        outline=(120, 120, 120),
        width=2,
    )


def plot_confusion(cm: np.ndarray, names: Sequence[str], path: Path, title: str) -> None:
    """计数 + 行归一化两个面板，画成一张 PNG（纯 Pillow 实现）。"""
    row_sums = cm.sum(axis=1, keepdims=True)
    rate = np.divide(
        cm,
        np.maximum(row_sums, 1),
        dtype=np.float64,
        out=np.zeros_like(cm, dtype=np.float64),
    )

    cell = 110
    k = len(names)
    left = int(cell * 2.0)  # 左边留给 "true: xxx"
    gap = int(cell * 2.0)  # 两个面板间距（要容下第二个面板的行标签）
    top = int(cell * 2.1)  # 上面留给标题和列标签
    bottom = int(cell * 0.8)

    width = left + k * cell + gap + k * cell + int(cell * 1.0)
    height = top + k * cell + bottom
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font_title = _load_font(int(cell * 0.42))
    font_note = _load_font(int(cell * 0.26))
    draw.text((int(cell * 0.25), int(cell * 0.2)), title, fill=(16, 16, 16), font=font_title)
    draw.text(
        (int(cell * 0.25), int(cell * 0.2) + int(cell * 0.5)),
        "rows = true label, columns = predicted label",
        fill=(90, 90, 90),
        font=font_note,
    )

    _draw_panel(draw, cm, names, (left, top), cell, "counts", "int", None)
    _draw_panel(
        draw,
        rate,
        names,
        (left + k * cell + gap, top),
        cell,
        "row-normalized",
        "pct",
        1.0,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def confusion_text(cm: np.ndarray, names: Sequence[str]) -> str:
    """纯文本混淆矩阵（终端输出 / txt 文件共用）。"""
    width = max(len("true\\pred"), *(len(n) for n in names)) + 2
    header = "true\\pred".ljust(width) + "".join(f"pred:{n}".rjust(width + 6) for n in names)
    header += "support".rjust(width + 6)
    lines = ["混淆矩阵（行 = 真实标签，列 = 预测标签）", header]
    for i, name in enumerate(names):
        row = f"true:{name}".ljust(width)
        row += "".join(str(int(v)).rjust(width + 6) for v in cm[i])
        row += str(int(cm[i].sum())).rjust(width + 6)
        lines.append(row)

    lines.append("")
    lines.append("按真实类归一化（每行和为 100%）")
    row_sums = cm.sum(axis=1, keepdims=True)
    rate = np.divide(
        cm,
        np.maximum(row_sums, 1),
        dtype=np.float64,
        out=np.zeros_like(cm, dtype=np.float64),
    )
    for i, name in enumerate(names):
        row = f"true:{name}".ljust(width)
        row += "".join(f"{v * 100:6.1f}%".rjust(width + 6) for v in rate[i])
        row += f"{100.0:6.1f}%".rjust(width + 6)
        lines.append(row)
    return "\n".join(lines)



def write_predictions(
    path: Path,
    filenames: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "true", "pred", "correct"]
            + [f"p_{n}" for n in names]
        )
        for name, t, p, prob in zip(filenames, y_true, y_pred, probs):
            writer.writerow(
                [name, names[t], names[p], int(t == p)]
                + [f"{v:.4f}" for v in prob]
            )


def print_worst(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    filenames: Sequence[str],
    names: Sequence[str],
    top_k: int,
) -> None:
    """打印被错分得最“自信”的样本，便于人工排查。"""
    wrong = np.where(y_true != y_pred)[0]
    if wrong.size == 0:
        print("\n验证集上没有错分样本。")
        return
    conf = probs[wrong, y_pred[wrong]]
    order = wrong[np.argsort(-conf)][:top_k]
    print(f"\n错分样本 {wrong.size} 张，最自信的 {min(top_k, wrong.size)} 张：")
    for idx in order:
        print(
            f"  {filenames[idx]:<16} true={names[y_true[idx]]:<4} "
            f"pred={names[y_pred[idx]]:<4} p(pred)={probs[idx, y_pred[idx]]:.3f}"
        )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载 checkpoint 评估猫狗分类结果（含混淆矩阵）"
    )
    parser.add_argument("--ckpt", required=True, help="ckpt 文件或 checkpoints 目录")
    parser.add_argument(
        "--model", default="auto",
        choices=["auto", "dnn", "cnn", "rnn", "cnn_attention", "vit"],
        help="模型类型，默认从 ckpt 超参自动推断",
    )
    parser.add_argument("--split", default="val", choices=["val", "train", "both"])
    parser.add_argument("--root", default=None, help="数据根目录，默认 data/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--image-size", type=int, default=None,
        help="预处理边长，默认从 ckpt 超参推断（DNN/RNN）或 192（CNN）",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-batches", type=int, default=None, help="只跑前 N 个 batch（冒烟测试）")
    parser.add_argument("--top-k-errors", type=int, default=10)
    parser.add_argument("--out", default=None, help="输出目录，默认 outputs/eval/<ckpt名>_<split>")
    parser.add_argument("--no-plot", action="store_true", help="不生成 PNG")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    ckpt_path = resolve_ckpt(Path(args.ckpt))
    ckpt = load_ckpt(ckpt_path)
    hparams = dict(ckpt.get("hyper_parameters") or {})
    state_dict = ckpt.get("state_dict") or {}

    model_name = (
        infer_model_name(hparams) if args.model == "auto" else args.model
    )
    print(f"checkpoint : {ckpt_path}")
    print(f"模型类型   : {model_name}（{'自动推断' if args.model == 'auto' else '手动指定'}）")

    model = build_model(model_name, hparams)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[警告] 权重不完全匹配: missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:5]:
            print(f"  missing 示例: {list(missing)[:5]}")
        if unexpected[:5]:
            print(f"  unexpected 示例: {list(unexpected)[:5]}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量     : {n_params / 1e6:.2f} M")

    device = torch.device(
        "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    print(f"设备       : {device}")

    # --- 数据 ---
    data_cfg = build_data_cfg(model_name, hparams, args.batch_size, args.image_size)
    dm_kwargs: dict[str, Any] = {"num_workers": args.num_workers}
    if args.root:
        dm_kwargs["root"] = args.root
    dm = CatDogDataModule(data_cfg, **dm_kwargs)
    dm.setup("validate")
    stats = dm.stats
    val_tf = build_pipeline(data_cfg.val, stats=stats)
    collate = make_collate_fn(data_cfg.variable_size)
    use_mask = bool(data_cfg.variable_size)

    def loader_for(split: str) -> DataLoader:
        if split == "val":
            dataset = dm._val_ds
        else:
            # 训练集用确定性（验证）管线，避免随机增强干扰指标
            dataset = CatDogDataset(split, root=dm.root, transform=val_tf)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            drop_last=False,
        )

    splits = ["val", "train"] if args.split == "both" else [args.split]
    out_dir = Path(args.out) if args.out else Path("outputs/eval") / f"{ckpt_path.stem}_{args.split}"

    summary: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "model": model_name,
        "num_parameters": int(n_params),
        "epoch": ckpt.get("epoch"),
        "splits": {},
    }

    for split in splits:
        loader = loader_for(split)
        print(f"\n{'=' * 60}\n[{split}] 样本数 {len(loader.dataset)}，batch={args.batch_size}")
        probs, y_true = predict(
            model, loader, device, use_mask=use_mask, limit_batches=args.limit_batches
        )
        y_pred = probs.argmax(axis=1)
        names = CLASSES[: probs.shape[1]]

        cm = confusion_matrix(y_true, y_pred, probs.shape[1])
        metrics = classification_metrics(cm, names)
        if probs.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_binary(y_true, probs[:, 1])

        print_confusion(cm, names)
        print_metrics(metrics)

        # 置信度分布
        max_prob = probs.max(axis=1)
        correct = y_true == y_pred
        metrics["confidence"] = {
            "mean_correct": round(float(max_prob[correct].mean()), 4) if correct.any() else None,
            "mean_wrong": round(float(max_prob[~correct].mean()), 4) if (~correct).any() else None,
        }
        print(
            f"\n平均置信度: 正确 {metrics['confidence']['mean_correct']}, "
            f"错误 {metrics['confidence']['mean_wrong']}"
        )

        filenames = [Path(p).name for p, _ in loader.dataset.samples]
        if args.limit_batches is not None:
            filenames = filenames[: len(y_true)]
        print_worst(y_true, y_pred, probs, filenames, names, args.top_k_errors)

        # 落盘
        split_dir = out_dir if len(splits) == 1 else out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        write_predictions(split_dir / "predictions.csv", filenames, y_true, y_pred, probs, names)
        with open(split_dir / "confusion_matrix.txt", "w", encoding="utf-8") as f:
            f.write(confusion_text(cm, names) + "\n")
        png_path = split_dir / "confusion_matrix.png"
        if not args.no_plot:
            try:
                plot_confusion(cm, names, png_path, title=f"{model_name} ({split})")
                print(f"\n混淆矩阵图: {png_path}")
            except Exception as exc:  # noqa: BLE001 - 画图失败不应中断评估
                print(f"\n[警告] 生成混淆矩阵图失败: {exc}（文本/CSV/JSON 仍然可用）")
        print(f"混淆矩阵文本: {split_dir / 'confusion_matrix.txt'}")
        print(f"逐样本预测: {split_dir / 'predictions.csv'}")

        summary["splits"][split] = {
            "confusion_matrix": cm.tolist(),
            "class_names": list(names),
            "metrics": metrics,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n指标汇总: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
