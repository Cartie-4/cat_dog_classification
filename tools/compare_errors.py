"""对比两个模型的**错分样本集合**：是不是同一批图在错。

用法::

    python tools/compare_errors.py \
        --a outputs/eval/cnn/predictions.csv cnn \
        --b outputs/eval/rnn/predictions.csv rnn \
        --out outputs/eval/error_overlap.json

输出：
- 两个模型各自的准确率 / 混淆矩阵
- 错分集合的大小、交集、Jaccard
- 独立性假设下的期望交集 + 超几何检验（错分是否显著相关）
- 按真实类别拆分的 "都错 / 只有 A 错 / 只有 B 错"
- 共同错分样本清单（供人工看图）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """读 predictions.csv，返回 {filename: {true, pred, correct}}。"""
    rows: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["filename"]] = {
                "true": row["true"],
                "pred": row["pred"],
                "correct": row["correct"] == "1",
            }
    return rows


def confusion(rows: dict[str, dict[str, Any]], classes: list[str]) -> list[list[int]]:
    idx = {c: i for i, c in enumerate(classes)}
    cm = [[0] * len(classes) for _ in classes]
    for r in rows.values():
        cm[idx[r["true"]]][idx[r["pred"]]] += 1
    return cm


def hypergeom_tail(n: int, k_success: int, draws: int, observed: int) -> float:
    """P(X >= observed)，X ~ Hypergeom(N=n, K=k_success, n=draws)。"""
    total = math.comb(n, draws)
    tail = 0
    for x in range(observed, min(k_success, draws) + 1):
        tail += math.comb(k_success, x) * math.comb(n - k_success, draws - x)
    return tail / total


def main() -> None:
    parser = argparse.ArgumentParser(description="对比两个模型的错分样本集合")
    parser.add_argument("--a", nargs=2, required=True, metavar=("CSV", "NAME"))
    parser.add_argument("--b", nargs=2, required=True, metavar=("CSV", "NAME"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--show", type=int, default=15, help="列出多少个共同错分样本")
    args = parser.parse_args()

    path_a, name_a = Path(args.a[0]), args.a[1]
    path_b, name_b = Path(args.b[0]), args.b[1]
    rows_a = load_predictions(path_a)
    rows_b = load_predictions(path_b)

    common = sorted(set(rows_a) & set(rows_b))
    if len(common) != len(rows_a) or len(common) != len(rows_b):
        print(
            f"[warn] 两个文件样本数不一致：{name_a}={len(rows_a)} "
            f"{name_b}={len(rows_b)} 交集={len(common)}，只比较交集部分"
        )
    n = len(common)
    classes = ["cat", "dog"]

    err_a = {f for f in common if not rows_a[f]["correct"]}
    err_b = {f for f in common if not rows_b[f]["correct"]}
    both = err_a & err_b
    only_a = err_a - err_b
    only_b = err_b - err_a
    correct_both = {f for f in common if rows_a[f]["correct"] and rows_b[f]["correct"]}

    acc_a = sum(rows_a[f]["correct"] for f in common) / n
    acc_b = sum(rows_b[f]["correct"] for f in common) / n

    expected = len(err_a) * len(err_b) / n
    p_value = hypergeom_tail(n, len(err_a), len(err_b), len(both))
    jaccard = len(both) / len(err_a | err_b) if (err_a | err_b) else float("nan")

    print("=" * 72)
    print(f"样本数 {n}（两个模型共同评估的图片）")
    print(f"{name_a:<12} 准确率 {acc_a * 100:6.2f}%   错分 {len(err_a):3d} 张")
    print(f"{name_b:<12} 准确率 {acc_b * 100:6.2f}%   错分 {len(err_b):3d} 张")
    print("-" * 72)
    print(f"混淆矩阵（行=真实 cat/dog，列=预测 cat/dog）")
    print(f"  {name_a:<12} {confusion({f: rows_a[f] for f in common}, classes)}")
    print(f"  {name_b:<12} {confusion({f: rows_b[f] for f in common}, classes)}")
    print("-" * 72)
    print(f"两个都分对        : {len(correct_both):3d}")
    print(f"两个都分错        : {len(both):3d}   <-- 交集")
    print(f"只有 {name_a:<10} 错 : {len(only_a):3d}")
    print(f"只有 {name_b:<10} 错 : {len(only_b):3d}")
    print("-" * 72)
    print(f"错分集合 Jaccard  : {jaccard:.3f}")
    print(f"独立假设下期望交集: {expected:.1f} 张")
    print(f"实际交集 / 期望   : {len(both) / expected:.2f} 倍" if expected else "")
    print(f"超几何检验 P(重叠 >= {len(both)}) = {p_value:.3e}")
    print("-" * 72)

    # 按真实类别拆分
    by_class: dict[str, Any] = {}
    for cls in classes:
        idx = [f for f in common if rows_a[f]["true"] == cls]
        ea = {f for f in idx if not rows_a[f]["correct"]}
        eb = {f for f in idx if not rows_b[f]["correct"]}
        by_class[cls] = {
            "n": len(idx),
            "err_a": len(ea),
            "err_b": len(eb),
            "both_err": len(ea & eb),
            "only_a": len(ea - eb),
            "only_b": len(eb - ea),
            "both_correct": len({f for f in idx if rows_a[f]["correct"] and rows_b[f]["correct"]}),
        }
        print(
            f"真实 {cls:<4} n={len(idx):3d} | 错: {name_a} {len(ea):3d}, "
            f"{name_b} {len(eb):3d} | 都错 {len(ea & eb):3d} | "
            f"只有 {name_a} 错 {len(ea - eb):3d} | 只有 {name_b} 错 {len(eb - ea):3d}"
        )

    # 方向是否一致：都错但预测方向相同 / 相反
    same_direction = sum(
        1 for f in both if rows_a[f]["pred"] == rows_b[f]["pred"]
    )
    print("-" * 72)
    print(
        f"共同错分里方向相同的: {same_direction}/{len(both)} "
        f"（同一张图、同样被误判成同一个类）"
    )
    print("=" * 72)

    if both:
        print(f"\n共同错分样本（前 {min(args.show, len(both))} 张）：")
        for f in sorted(both)[: args.show]:
            print(
                f"  {f:<16} true={rows_a[f]['true']:<4} "
                f"{name_a}→{rows_a[f]['pred']:<4} {name_b}→{rows_b[f]['pred']}"
            )

    result = {
        "n": n,
        "models": {name_a: {"csv": str(path_a), "accuracy": acc_a, "errors": len(err_a)},
                   name_b: {"csv": str(path_b), "accuracy": acc_b, "errors": len(err_b)}},
        "confusion": {
            name_a: confusion({f: rows_a[f] for f in common}, classes),
            name_b: confusion({f: rows_b[f] for f in common}, classes),
        },
        "overlap": {
            "both_correct": len(correct_both),
            "both_wrong": len(both),
            "only_a_wrong": len(only_a),
            "only_b_wrong": len(only_b),
            "jaccard": jaccard,
            "expected_independent": expected,
            "ratio_over_expected": len(both) / expected if expected else None,
            "hypergeom_p": p_value,
            "same_direction": same_direction,
        },
        "by_true_class": by_class,
        "shared_errors": sorted(both),
        "only_a_errors": sorted(only_a),
        "only_b_errors": sorted(only_b),
    }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
