"""定量测量"数据够不够"：学习曲线 + 幂律拟合。

原理
----
模型精度随训练样本数 N 大致服从幂律：

    acc(N) = a - b * N^(-alpha)

- ``a``     ：N -> 无穷 时的精度天花板（外推值）
- ``b``     ：当前离天花板有多远
- ``alpha`` ：数据效率指数，越大说明"加数据"收敛得越快

由拟合结果可以直接给出三个可解释的指标：

1. **数据充足度** = acc(N_max) / a
   越接近 1 说明当前数据已接近该模型上限，再加数据收益小；
   明显小于 1 说明数据是瓶颈。
2. **数据缺口** = a - acc(N_max)
   距离自己天花板还有多少个百分点。
3. **达到目标精度所需的样本数**
   例如"要到 90%，按拟合曲线需要多少张图"。

做法：按分层比例抽取训练集子集（保持猫狗 1:1），在每个子集上训练同一个配置，
记录验证精度，最后对 (N, acc) 做幂律最小二乘拟合。

用法::

    python tools/data_scaling.py --model vit --fractions 0.25 0.5 1.0 --epochs 40
    python tools/data_scaling.py --model cnn_attention --fractions 0.25 0.5 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from catdog.train import CONFIG_CLASSES, build


def subsample(dataset, fraction: float, seed: int = 0):
    """按类别分层抽取 ``fraction`` 比例的样本，保持猫狗比例不变。"""
    if fraction >= 1.0:
        return dataset
    by_label: dict[int, list] = {}
    for item in dataset.samples:
        by_label.setdefault(item[1], []).append(item)
    rng = random.Random(seed)
    keep = []
    for label, items in by_label.items():
        k = max(1, int(round(len(items) * fraction)))
        keep.extend(rng.sample(items, k))
    keep.sort(key=lambda x: str(x[0]))
    dataset.samples = keep
    return dataset


def fit_power_law(ns: list[float], accs: list[float]) -> dict[str, float] | None:
    """拟合 acc = a - b * N^(-alpha)，返回 a / b / alpha。

    做法：固定 alpha 网格搜索，对每个 alpha 用线性最小二乘解 (a, b)，
    取残差平方和最小的那组。这样比三参数非线性拟合稳定得多。
    """
    if len(ns) < 3:
        return None
    n = np.asarray(ns, dtype=np.float64)
    y = np.asarray(accs, dtype=np.float64)
    best = None
    for alpha in np.linspace(0.05, 2.0, 196):
        x = n ** (-alpha)
        # y = a - b*x  ->  线性最小二乘
        A = np.stack([np.ones_like(x), -x], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = A @ coef
        sse = float(((y - pred) ** 2).sum())
        if best is None or sse < best["sse"]:
            best = {
                "a": float(coef[0]),
                "b": float(coef[1]),
                "alpha": float(alpha),
                "sse": sse,
            }
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="学习曲线：定量测量数据充足程度")
    p.add_argument("--model", required=True,
                   choices=["dnn", "cnn", "rnn", "cnn_attention", "vit"])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--epochs", type=int, default=40,
                   help="固定 epoch 数（为了各子集/各模型之间可比，不早停）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default="outputs/scaling")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    jsonl = out_dir / f"{args.model}{suffix}_scaling.jsonl"

    cfg = CONFIG_CLASSES[args.model]()
    results = []
    print(f"[scaling] model={args.model} fractions={args.fractions} epochs={args.epochs}")
    for frac in args.fractions:
        pl.seed_everything(args.seed, workers=True)
        model, dm = build(args.model)

        # 先让 DataModule 建好训练/验证集，再对训练集做分层子抽样
        dm.setup("fit")
        n_full = len(dm._train_ds)
        dm._train_ds = subsample(dm._train_ds, frac, seed=args.seed)
        n = len(dm._train_ds)

        recorder_records: list[dict] = []

        class Rec(pl.Callback):
            def on_validation_end(self, trainer, pl_module):
                m = trainer.callback_metrics
                if "val_acc" in m and "val_loss" in m:
                    ep = int(trainer.current_epoch)
                    if not recorder_records or recorder_records[-1]["epoch"] != ep:
                        recorder_records.append(
                            {
                                "epoch": ep,
                                "val_acc": float(m["val_acc"]),
                                "val_loss": float(m["val_loss"]),
                            }
                        )

        trainer = pl.Trainer(
            accelerator="auto", devices=1, precision="16-mixed",
            max_epochs=args.epochs, logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0, log_every_n_steps=50,
            callbacks=[Rec()],
        )
        t0 = time.perf_counter()
        trainer.fit(model, datamodule=dm)
        secs = time.perf_counter() - t0

        best = max(r["val_acc"] for r in recorder_records)
        last = recorder_records[-1]["val_acc"]
        rec = {
            "model": args.model, "fraction": frac, "n_train": n, "n_full": n_full,
            "epochs": args.epochs, "best_val_acc": best, "last_val_acc": last,
            "seconds": round(secs, 1), "records": recorder_records,
        }
        results.append(rec)
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"  frac={frac:<5} N={n:<5} best={best:.4f} last={last:.4f} "
            f"({secs:.0f}s, {secs / args.epochs:.1f}s/ep)"
        )
        del trainer, model, dm
        torch.cuda.empty_cache()

    # 幂律拟合
    ns = [r["n_train"] for r in results]
    accs = [r["best_val_acc"] for r in results]
    fit = fit_power_law(ns, accs)
    print()
    if fit:
        ceiling = fit["a"]
        cur = max(accs)
        ratio = cur / ceiling if ceiling else float("nan")
        print(f"[scaling] {args.model}: 天花板 a={ceiling * 100:.2f}%  "
              f"alpha={fit['alpha']:.3f}  b={fit['b']:.3f}")
        print(f"  当前(N={max(ns)})最好 {cur * 100:.2f}%  "
              f"数据充足度 {ratio * 100:.1f}%  数据缺口 {(ceiling - cur) * 100:.2f} pt")
        for target in (0.85, 0.90):
            if ceiling > target:
                need = (fit["b"] / (ceiling - target)) ** (1 / fit["alpha"])
                print(f"  要达到 {target * 100:.0f}%，按拟合外推需要约 {need:,.0f} 张图")
        with open(out_dir / f"{args.model}{suffix}_fit.json", "w", encoding="utf-8") as f:
            json.dump(
                {"model": args.model, "fit": fit, "points": results_slim(results)},
                f, ensure_ascii=False, indent=2,
            )
    else:
        print("[scaling] 点数不足（至少 3 个 fraction）无法拟合幂律")


def results_slim(results):
    return [
        {"n_train": r["n_train"], "best_val_acc": r["best_val_acc"]} for r in results
    ]


if __name__ == "__main__":
    main()
