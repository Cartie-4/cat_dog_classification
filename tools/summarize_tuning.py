"""汇总 ``catdog.tune`` 的搜索结果。

用法::

    python tools/summarize_tuning.py                 # 默认读 outputs/tuning
    python tools/summarize_tuning.py --dir outputs/tuning --top 6

读取 ``<model>_trials.jsonl`` / ``<model>_baseline_trials.jsonl``，
输出每个模型「基线 vs 搜索最优」的对比表（Markdown + 终端）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

MODELS = ["dnn", "cnn", "rnn", "cnn_attention", "vit"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def sort_key(result: dict[str, Any]) -> tuple[float, float]:
    acc = result.get("best_val_acc")
    loss = result.get("best_val_loss")
    if acc is None:
        return (float("-inf"), float("inf"))
    return (acc, -(loss if loss is not None else float("inf")))


def epoch_curve(result: dict[str, Any]) -> str:
    records = result.get("records") or []
    if not records:
        return "-"
    return " -> ".join(f"{r['val_acc']:.3f}" for r in records)


def fmt_params(params: dict[str, Any]) -> str:
    if not params:
        return "(默认)" 
    return ", ".join(f"{k}={v}" for k, v in params.items())


def summarize_model(model: str, directory: Path, top: int) -> list[str]:
    baseline = load_jsonl(directory / f"{model}_baseline_trials.jsonl")
    search = load_jsonl(directory / f"{model}_trials.jsonl")
    lines: list[str] = []

    if not baseline and not search:
        return lines

    lines.append(f"## {model}")
    lines.append("")

    base = baseline[0] if baseline else None
    best_search = max(search, key=sort_key) if search else None
    if base:
        acc = base.get("best_val_acc")
        lines.append(
            f"- 基线（默认参数，{base.get('epochs')} epoch）: "
            f"val_acc = {acc if acc is None else f'{acc * 100:.2f}%'}，"
            f"耗时 {base.get('seconds')}s"
        )
    if best_search:
        acc = best_search.get("best_val_acc")
        lines.append(
            f"- 搜索最优（{best_search.get('epochs')} epoch）: "
            f"val_acc = {acc if acc is None else f'{acc * 100:.2f}%'}，"
            f"耗时 {best_search.get('seconds')}s"
        )
        if base and base.get("best_val_acc") and best_search.get("best_val_acc"):
            delta = (best_search["best_val_acc"] - base["best_val_acc"]) * 100
            lines.append(f"- 提升: {delta:+.2f} 个百分点")
    lines.append("")
    lines.append(f"搜索共 {len(search)} 组，按 val_acc 排序前 {min(top, len(search))} 组：")
    lines.append("")
    lines.append("| # | val_acc | val_loss | 秒 | 每 epoch val_acc | 参数 |")
    lines.append("|---|---|---|---|---|---|")
    ranked = sorted([r for r in search if r.get("best_val_acc") is not None], key=sort_key, reverse=True)
    for i, res in enumerate(ranked[:top], 1):
        lines.append(
            f"| {i} | {res['best_val_acc'] * 100:.2f}% | "
            f"{res['best_val_loss']:.4f} | {res['seconds']:.0f} | "
            f"{epoch_curve(res)} | {fmt_params(res['params'])} |"
        )
    failed = [r for r in search if r.get("status") != "ok"]
    if failed:
        lines.append("")
        lines.append(f"失败/异常 {len(failed)} 组：" + "；".join(
            f"[{r['status']}] {fmt_params(r['params'])}" for r in failed
        ))
    lines.append("")
    return lines


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="汇总调参结果")
    parser.add_argument("--dir", default="outputs/tuning")
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--out", default=None, help="同时写出 Markdown 文件")
    args = parser.parse_args(argv)

    directory = Path(args.dir)
    lines: list[str] = ["# 调参结果汇总", ""]
    for model in MODELS:
        lines.extend(summarize_model(model, directory, args.top))

    text = "\n".join(lines)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"\n[summarize] 已写入 {out_path}")


if __name__ == "__main__":
    main()
