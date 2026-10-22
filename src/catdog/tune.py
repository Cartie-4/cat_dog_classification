"""超参数搜索（随机搜索 + 时间预算）。

用法::

    # 基线：不覆盖任何超参，跑一次记录默认 cfg 的表现
    python -m catdog.tune --model cnn --epochs 8 --pin '{}' --tag baseline

    # 随机搜索：最多 12 组，或总耗时不超过 20 分钟
    python -m catdog.tune --model cnn --trials 12 --epochs 8 --time-budget-min 20

    # 复跑某一组具体参数（Windows 下建议用 --params-file，避免引号转义踩坑）
    python -m catdog.tune --model cnn --epochs 8 --pin '{"lr": 0.003, "weight_decay": 0.0001, "dropout": 0.2}'
    python -m catdog.tune --model cnn --epochs 16 --params-file best.json --save-ckpt --tag final

设计说明
--------
- **代理预算**：搜索阶段只跑少量 epoch（默认 8），用「验证集上最好的
  ``val_acc``」排序。同一次搜索里所有 trial 的 epoch 数、数据管线、随机种子
  完全一致，因此可以公平比较相对优劣。
- **不落 checkpoint / 不写 TensorBoard**：搜索只关心指标，省掉 I/O。
- **结果增量落盘**：每次 trial 结束立刻追加一行到
  ``<out>/<model>_trials.jsonl``，中途中断也不丢结果。
- 搜索用的超参范围见 ``SEARCH_SPACES``；写回 ``cfg.py`` 的默认值应取
  ``best.json`` 里的推荐配置。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from catdog.data.cfg import DEFAULT_IMAGE_SIZE, image_preset
from catdog.models.CNN import CNNConfig
from catdog.models.CNN_attention import CNNAttentionConfig

# ``_attention_data`` 是 CNN_attention 的专用预处理工厂；这里直接复用，
# 保证搜索时的数据管线和正式训练完全一致。
from catdog.models.CNN_attention.cfg import _attention_data
from catdog.models.DNN import DNNConfig
from catdog.models.RNN import RNNConfig
from catdog.models.ViT import ViTConfig
from catdog.train import build

CONFIGS = {
    "dnn": DNNConfig,
    "cnn": CNNConfig,
    "rnn": RNNConfig,
    "cnn_attention": CNNAttentionConfig,
    "vit": ViTConfig,
}

# ---------------------------------------------------------------------------
# 搜索空间
# ---------------------------------------------------------------------------
# 每项: {"type": "log" | "linear" | "choice", ...}
# - log:    对数均匀采样（适合 lr / weight_decay）
# - linear: 均匀采样（适合 dropout）
# - choice: 离散候选
#
# 约定：
# - ``batch_size`` 会被写进 ``cfg.data``；
# - ``image_size``（可选）同时决定预处理边长和模型输入维度：
#   DNN 的 ``in_dim``、RNN 的 ``pic_size`` 会跟着变；CNN 只影响预处理。
SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "dnn": {
        "lr": {"type": "log", "lo": 1e-5, "hi": 3e-3},
        "weight_decay": {"type": "log", "lo": 1e-6, "hi": 1e-1},
        "dropout": {"type": "linear", "lo": 0.0, "hi": 0.5},
        "hidden_dim": {"type": "choice", "values": [128, 256, 512]},
        "bottleneck_dim": {"type": "choice", "values": [64, 128, 256]},
        "batch_size": {"type": "choice", "values": [32, 64, 128]},
    },
    "cnn": {
        "lr": {"type": "log", "lo": 1e-4, "hi": 2e-2},
        "weight_decay": {"type": "log", "lo": 1e-6, "hi": 1e-2},
        "dropout": {"type": "linear", "lo": 0.0, "hi": 0.5},
        "base_width": {"type": "choice", "values": [32, 64, 96]},
        "batch_size": {"type": "choice", "values": [32, 64, 128]},
    },
    "rnn": {
        "lr": {"type": "log", "lo": 3e-5, "hi": 5e-3},
        "weight_decay": {"type": "log", "lo": 1e-6, "hi": 1e-2},
        "dropout": {"type": "linear", "lo": 0.0, "hi": 0.5},
        "rnn_hidden": {"type": "choice", "values": [128, 256, 384]},
        "hidden_dim": {"type": "choice", "values": [128, 256]},
        "num_layers": {"type": "choice", "values": [1, 2, 3]},
        "batch_size": {"type": "choice", "values": [32, 64]},
    },
    # CNN_attention 保留原图尺寸，显存/耗时对 batch 和 base_width 很敏感，
    # 所以候选值都比其他模型更保守。
    "cnn_attention": {
        "lr": {"type": "log", "lo": 1e-4, "hi": 1e-2},
        "weight_decay": {"type": "log", "lo": 1e-6, "hi": 1e-2},
        "dropout": {"type": "linear", "lo": 0.0, "hi": 0.4},
        "fnn_hidden": {"type": "choice", "values": [128, 256, 512]},
        "base_width": {"type": "choice", "values": [32, 64]},
        "batch_size": {"type": "choice", "values": [4, 8, 16]},
    },
    # ViT 没有卷积先验，对小数据集很敏感：lr / weight_decay / dropout 是主要旋钮。
    # embed_dim 的候选都能被 num_heads=4 整除；patch_size=32 时 token 数从 144 降到 36。
    "vit": {
        "lr": {"type": "log", "lo": 1e-5, "hi": 1e-3},
        "weight_decay": {"type": "log", "lo": 1e-4, "hi": 5e-1},
        "dropout": {"type": "linear", "lo": 0.0, "hi": 0.3},
        "embed_dim": {"type": "choice", "values": [128, 192, 256]},
        "depth": {"type": "choice", "values": [4, 6, 8]},
        "patch_size": {"type": "choice", "values": [16, 32]},
        "batch_size": {"type": "choice", "values": [32, 64]},
    },
}


def sample_params(model: str, rng: random.Random) -> dict[str, Any]:
    """从搜索空间里采一组超参。"""
    params: dict[str, Any] = {}
    for name, spec in SEARCH_SPACES[model].items():
        kind = spec["type"]
        if kind == "choice":
            params[name] = rng.choice(spec["values"])
        elif kind == "log":
            lo, hi = math.log10(spec["lo"]), math.log10(spec["hi"])
            params[name] = float(f"{10 ** rng.uniform(lo, hi):.3g}")
        elif kind == "linear":
            params[name] = round(rng.uniform(spec["lo"], spec["hi"]), 3)
        else:  # pragma: no cover - 防御性分支
            raise ValueError(f"未知采样类型: {kind}")
    return params


def make_cfg(model: str, params: dict[str, Any]):
    """把采样到的超参套到 dataclass 配置上。

    ``batch_size`` / ``image_size`` 不在模型配置里，而在 ``cfg.data`` 里，需要
    单独重建预处理；DNN 的 ``in_dim``、RNN 的 ``pic_size`` 还要跟着尺寸一起改。
    """
    cfg_cls = CONFIGS[model]
    cfg = cfg_cls()
    scalar_keys = {f.name for f in fields(cfg)}
    overrides = {
        k: v for k, v in params.items() if k in scalar_keys
    }
    image_size = params.get("image_size")
    if image_size is not None:
        image_size = int(image_size)
        # 输入维度必须和预处理尺寸一致，否则 Linear/卷积形状对不上。
        if model == "dnn":
            overrides["in_dim"] = image_size * image_size * 3
        elif model == "rnn":
            overrides["pic_size"] = (image_size, image_size)
            patch = int(params.get("patch_size", cfg.patch_size))
            if image_size % patch != 0:
                raise ValueError(
                    f"image_size={image_size} 必须能被 patch_size={patch} 整除"
                )
        elif model == "vit":
            # ViT 的 image_size 决定位置编码长度，必须跟着预处理一起改
            overrides["image_size"] = image_size
    if overrides:
        cfg = replace(cfg, **overrides)

    if "batch_size" in params or image_size is not None:
        batch_size = int(params.get("batch_size", cfg.data.batch_size))
        if model == "cnn_attention":
            cfg = replace(cfg, data=_attention_data(batch_size))
        else:
            cfg = replace(
                cfg,
                data=image_preset(
                    image_size if image_size is not None else DEFAULT_IMAGE_SIZE,
                    batch_size,
                ),
            )
    return cfg


class _MetricsRecorder(pl.Callback):
    """每个 epoch 记录验证指标（搜索只需要这些）。

    .. note::
       必须用 ``on_validation_end`` 而不是 ``on_validation_epoch_end``：
       PL 在 ``on_validation_epoch_end`` 之后才把 ``val_*`` 写进
       ``trainer.callback_metrics``（见 ``_EvaluationLoop.on_run_end``）。
    """

    def __init__(self, print_every: int = 0) -> None:
        super().__init__()
        self.records: list[dict[str, float]] = []
        self._epoch_start: Optional[float] = None
        self.print_every = print_every

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # noqa: D102
        self._epoch_start = time.perf_counter()

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: D102
        metrics = trainer.callback_metrics
        if "val_acc" not in metrics or "val_loss" not in metrics:
            return
        epoch = int(trainer.current_epoch)
        # 去重：validate 阶段可能再次触发（搜索只跑 fit，这里只是防御）。
        if self.records and self.records[-1]["epoch"] == epoch:
            return
        secs = (
            time.perf_counter() - self._epoch_start
            if self._epoch_start is not None
            else 0.0
        )
        acc = float(metrics["val_acc"])
        loss = float(metrics["val_loss"])
        self.records.append(
            {
                "epoch": epoch,
                "val_acc": acc,
                "val_loss": loss,
                "train_seconds": round(secs, 1),
            }
        )
        # 可选：每 N 轮打一行，方便在长时间搜索里盯平台期
        if self.print_every and (epoch + 1) % self.print_every == 0:
            best = max(r["val_acc"] for r in self.records)
            print(
                f"    ep{epoch + 1:>3}  val_acc={acc:.4f}  val_loss={loss:.4f}"
                f"  best={best:.4f}  ({secs:.1f}s/ep)",
                flush=True,
            )


def _build_tuning_trainer(
    max_epochs: int,
    accelerator: str,
    devices: int,
    precision: str,
    callbacks: Optional[list] = None,
    enable_checkpointing: bool = False,
    early_stop_patience: int = 0,
    fast_dev_run: bool = False,
) -> pl.Trainer:
    """搜索专用 Trainer：默认不落盘、不打日志、不打印。

    ``enable_checkpointing=True`` 时需要外部传入 ``ModelCheckpoint`` 回调，
    用于把最优权重留下来给 ``catdog.evaluate`` 做混淆矩阵。

    ``early_stop_patience > 0`` 时启用 EarlyStopping（monitor=val_loss, mode=min）：
    这样每组超参都跑到**自己的收敛点**再比较，避免"固定 epoch 数"把
    lr 和训练预算混在一起（lr 大的配置几十轮就收敛，lr 小的要几百轮）。
    """
    hooks = callbacks or []
    if early_stop_patience > 0:
        hooks = hooks + [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=early_stop_patience,
                verbose=False,
            )
        ]
    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,  # 省掉训练前那一遍完整验证
        log_every_n_steps=50,
        callbacks=hooks,
        fast_dev_run=fast_dev_run,
    )


def run_trial(
    model: str,
    params: dict[str, Any],
    epochs: int,
    accelerator: str = "auto",
    devices: int = 1,
    precision: str = "16-mixed",
    num_workers: int = 0,
    seed: int = 42,
    stats_max_samples: Optional[int] = None,
    ckpt_dir: Optional[Path] = None,
    early_stop_patience: int = 0,
    progress_every: int = 0,
    fast_dev_run: bool = False,
) -> dict[str, Any]:
    """跑一组超参，返回指标汇总。

    ``ckpt_dir`` 非空时按 ``val_loss`` 保存最优权重（供 ``catdog.evaluate`` 使用）。
    ``early_stop_patience > 0`` 时每组超参训练到自己的早停点（``epochs`` 只是上限）。
    ``progress_every > 0`` 时每 N 轮打印一行验证指标，便于盯平台期。
    """
    pl.seed_everything(seed, workers=True)

    cfg = make_cfg(model, params)
    module, dm = build(
        model,
        num_workers=num_workers,
        stats_max_samples=stats_max_samples,
        cfg=cfg,
    )
    recorder = _MetricsRecorder(print_every=progress_every)
    callbacks: list = [recorder]
    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_weights_only=True,
                filename=f"{model}-epoch={{epoch:02d}}-val_loss={{val_loss:.4f}}",
                auto_insert_metric_name=False,
            )
        )
    trainer = _build_tuning_trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        enable_checkpointing=ckpt_dir is not None,
        early_stop_patience=early_stop_patience,
        fast_dev_run=fast_dev_run,
    )

    status = "ok"
    error = None
    started = time.perf_counter()
    try:
        trainer.fit(module, datamodule=dm)
    except RuntimeError as exc:  # 显存不足 / 数值爆炸
        if "out of memory" in str(exc).lower():
            status = "oom"
            error = "CUDA out of memory"
            torch.cuda.empty_cache()
        else:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - 搜索不应因单次失败中断
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started

    records = recorder.records
    vals = [r["val_acc"] for r in records if not math.isnan(r["val_acc"])]
    losses = [r["val_loss"] for r in records if not math.isnan(r["val_loss"])]
    if status == "ok" and not vals:
        status = "diverged"  # 训练跑完但没有有效指标（NaN）

    result: dict[str, Any] = {
        "model": model,
        "params": params,
        "status": status,
        "error": error,
        "epochs": epochs,
        "epochs_run": len(records),          # 实际跑了多少轮（早停时可能远小于 epochs）
        "early_stop_patience": early_stop_patience,
        "seconds": round(elapsed, 1),
        "sec_per_epoch": round(elapsed / max(1, len(records)), 1),
        "best_val_acc": max(vals) if vals else None,
        "last_val_acc": vals[-1] if vals else None,
        "best_val_loss": min(losses) if losses else None,
        "records": records,
    }

    # 立即释放，避免连续搜索时显存累积。
    del trainer, module, dm
    torch.cuda.empty_cache()
    return result


def _sort_key(result: dict[str, Any]) -> tuple[float, float]:
    """排序：验证准确率优先，其次验证损失。"""
    acc = result.get("best_val_acc")
    loss = result.get("best_val_loss")
    if acc is None:
        return (float("-inf"), float("inf"))
    return (acc, -(loss if loss is not None else float("inf")))


def _print_summary(results: list[dict[str, Any]]) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    ok.sort(key=_sort_key, reverse=True)
    print("\n" + "=" * 100)
    print(f"{'rank':<5}{'val_acc':<10}{'val_loss':<10}{'ep':<6}{'sec':<8}{'status':<9}params")
    print("-" * 100)
    for rank, res in enumerate(ok, 1):
        print(
            f"{rank:<5}{res['best_val_acc']:<10.4f}{res['best_val_loss']:<10.4f}"
            f"{res.get('epochs_run', 0):<6}{res['seconds']:<8.0f}{res['status']:<9}"
            f"{json.dumps(res['params'], ensure_ascii=False)}"
        )
    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        print("-" * 90)
        for res in failed:
            print(f"[{res['status']}] {json.dumps(res['params'])} -> {res['error']}")
    print("=" * 90 + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cat vs Dog 超参搜索")
    parser.add_argument(
        "--model", required=True,
        choices=["dnn", "cnn", "rnn", "cnn_attention", "vit"],
    )
    parser.add_argument("--trials", type=int, default=10, help="随机搜索的 trial 数")
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="每组超参的代理 epoch 数；配合 --early-stop-patience 时它是上限",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help=(
            ">0 时启用 EarlyStopping(val_loss, patience=N)：每组超参训练到自己的收敛点，"
            "这样比较的是「各自收敛后的精度」，不再受固定 epoch 数与 lr 的耦合影响。"
            "0 表示不早停、固定跑 --epochs 轮。"
        ),
    )
    parser.add_argument(
        "--time-budget-min", type=float, default=0.0, help="总耗时上限（分钟），0 表示不限"
    )
    parser.add_argument(
        "--pin",
        default=None,
        help='固定参数（JSON），例如 \'{"lr": 0.001}\'；给空对象即跑基线',
    )
    parser.add_argument(
        "--params-file",
        default=None,
        help="从 JSON 文件读取固定参数（推荐：避免命令行引号转义问题）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stats-max-samples", type=int, default=None)
    parser.add_argument("--out", default="outputs/tuning", help="结果目录")
    parser.add_argument("--tag", default=None, help="结果文件名后缀（如 baseline）")
    parser.add_argument(
        "--save-ckpt",
        action="store_true",
        help="保存验证集最优权重到 <out>/ckpt_<tag>/，供 catdog.evaluate 使用",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="每 N 轮打印一行验证指标（0 = 不打印）；长跑时用来盯平台期",
    )
    parser.add_argument(
        "--fast-dev-run", action="store_true", help="冒烟测试：1 个 batch"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    jsonl_path = out_dir / f"{args.model}{suffix}_trials.jsonl"
    best_path = out_dir / f"{args.model}{suffix}_best.json"

    base_seed = args.seed
    rng = random.Random(base_seed)
    pinned: Optional[dict[str, Any]] = None
    if args.params_file:
        # utf-8-sig：兼容 Windows PowerShell 写出的带 BOM 的 UTF-8
        pinned = json.loads(
            Path(args.params_file).read_text(encoding="utf-8-sig")
        )
    elif args.pin:
        pinned = json.loads(args.pin)

    # 固定参数时只跑一次，不覆盖未指定的字段。
    n_trials = 1 if pinned is not None else args.trials
    budget = args.time_budget_min * 60 if args.time_budget_min > 0 else float("inf")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[tune] model={args.model} trials={n_trials} epochs={args.epochs} "
        f"device={device} budget={args.time_budget_min or 'unlimited'}min",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for trial in range(n_trials):
        if time.perf_counter() - started > budget:
            print(f"[tune] 时间预算用尽，提前结束（已完成 {trial} 组）", flush=True)
            break

        params = pinned if pinned is not None else sample_params(args.model, rng)
        print(
            f"[tune] trial {trial + 1}/{n_trials} {json.dumps(params, ensure_ascii=False)}",
            flush=True,
        )

        result = run_trial(
            args.model,
            params,
            epochs=args.epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            num_workers=args.num_workers,
            seed=base_seed,
            stats_max_samples=args.stats_max_samples,
            ckpt_dir=(
                # 目录名带上模型名，避免同 tag 的多个模型互相覆盖
                out_dir / f"ckpt_{args.model}_{args.tag or f'trial{trial}'}"
                if args.save_ckpt
                else None
            ),
            early_stop_patience=args.early_stop_patience,
            progress_every=args.progress_every,
            fast_dev_run=args.fast_dev_run,
        )
        result["trial"] = trial
        results.append(result)

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        acc = result["best_val_acc"]
        print(
            f"[tune]   -> status={result['status']} "
            f"best_val_acc={acc if acc is None else f'{acc:.4f}'} "
            f"sec={result['seconds']}",
            flush=True,
        )

        # 随时把当前最优写出来，中断也有结果。
        ok = [r for r in results if r["status"] == "ok"]
        if ok:
            best = max(ok, key=_sort_key)
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": args.model,
                        "epochs": args.epochs,
                        "best_val_acc": best["best_val_acc"],
                        "best_val_loss": best["best_val_loss"],
                        "params": best["params"],
                        "cfg": _cfg_to_json(make_cfg(args.model, best["params"])),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    _print_summary(results)
    print(f"[tune] 结果已写入 {jsonl_path}", flush=True)
    total = time.perf_counter() - started
    print(f"[tune] 总耗时 {total / 60:.1f} 分钟", flush=True)


def _cfg_to_json(cfg) -> dict[str, Any]:
    """把配置对象转成可 JSON 序列化的 dict（便于人工核对）。"""
    data = cfg.__dict__.copy()
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key == "data":
            out["data"] = {
                "variable_size": value.variable_size,
                "batch_size": value.batch_size,
                "train": [t.to_dict() for t in value.train],
                "val": [t.to_dict() for t in value.val],
            }
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    main()
