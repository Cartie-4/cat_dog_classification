"""分析"训练到早停"的搜索结果：排名、收敛轮数、以及 lr 与收敛轮数的对应关系。

用法::

    python tools/analyze_search.py --dir outputs/tuning --suffix es_search
    python tools/analyze_search.py --model vit --out docs/search_summary.md

输出内容
--------
- 每组超参的 best val_acc / val_loss / **实际收敛轮数(epochs_run)** / 耗时
- 该组在**第几轮**达到最高准确率（"到达平台期的轮数"）
- lr 与 epochs_run 的对应关系（用于说明"epoch 数要和学习率对应"）
- 前若干名的曲线尾部，判断是否真的收敛
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODELS = ["dnn", "cnn", "rnn", "cnn_attention", "vit"]


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def argmax_epoch(records: list[dict[str, Any]]) -> int | None:
    """best val_acc 出现在第几轮。"""
    if not records:
        return None
    best = max(r["val_acc"] for r in records)
    for r in records:
        if r["val_acc"] == best:
            return int(r["epoch"]) + 1
    return None


def tail(values: list[float], k: int = 8) -> str:
    return " ".join(f"{v:.3f}" for v in values[-k:])


def summarize(model: str, rows: list[dict[str, Any]], top: int) -> list[str]:
    rows = [r for r in rows if r.get("best_val_acc") is not None]
    if not rows:
        return []
    rows.sort(key=lambda r: (r["best_val_acc"], -(r["best_val_loss"] or 9)), reverse=True)

    out = [f"## {model}", ""]
    out.append(
        f"共 {len(rows)} 组，全部训练到早停（cap "
        f"{rows[0].get('epochs', '?')} epoch，patience {rows[0].get('early_stop_patience', '?')}）"
    )
    out.append("")
    out.append("| # | val_acc | val_loss | 收敛轮数 | 峰值轮 | 秒 | 参数 |")
    out.append("|---|---|---|---|---|---|---|")

    for i, r in enumerate(rows[:top], 1):
        peak = argmax_epoch(r.get("records") or [])
        lr = r["params"].get("lr")
        lr_s = f"lr={lr:g}" if isinstance(lr, float) else f"lr={lr}"
        others = ", ".join(
            f"{k}={v}" for k, v in sorted(r["params"].items()) if k != "lr"
        )
        out.append(
            f"| {i} | {r['best_val_acc'] * 100:.2f}% | {r['best_val_loss']:.4f} | "
            f"{r.get('epochs_run', '?')} | {peak} | {r['seconds']:.0f} | "
            f"{lr_s}, {others} |"
        )

    # lr vs 收敛轮数
    out.append("")
    out.append("### lr 与收敛轮数的对应关系（epoch 数确实要跟着 lr 走）")
    out.append("")
    out.append("| lr | 收敛轮数 | val_acc |")
    out.append("|---|---|---|")
    for r in sorted(rows, key=lambda r: r["params"].get("lr", 0)):
        out.append(
            f"| {r['params'].get('lr'):g} | {r.get('epochs_run', '?')} | "
            f"{r['best_val_acc'] * 100:.2f}% |"
        )

    # 曲线尾部
    out.append("")
    out.append("### 曲线尾部（最后 8 轮 val_acc，看是否真的进平台）")
    out.append("")
    for i, r in enumerate(rows[: min(3, len(rows))], 1):
        recs = [x["val_acc"] for x in (r.get("records") or [])]
        out.append(f"- #{i}: {tail(recs)}")
    out.append("")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="分析早停式搜索结果")
    p.add_argument("--dir", default="outputs/tuning")
    p.add_argument("--suffix", default="es_search")
    p.add_argument("--model", default=None, help="只分析一个模型")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    directory = Path(args.dir)
    models = [args.model] if args.model else MODELS
    lines: list[str] = ["# 早停式搜索结果", ""]
    for m in models:
        rows = load(directory / f"{m}_{args.suffix}_trials.jsonl")
        lines.extend(summarize(m, rows, args.top))

    text = "\n".join(lines)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"\n[analyze] 已写入 {out}")


if __name__ == "__main__":
    main()
