"""从搜索结果里取每个模型的前 K 组参数，输出可直接喂给

    python -m catdog.tune --pin '<json>'

的 JSON 行，便于做「第二阶段：更长预算精选」。

用法::

    python tools/top_params.py --model dnn --top 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def sort_key(result: dict[str, Any]) -> tuple[float, float]:
    acc = result.get("best_val_acc")
    loss = result.get("best_val_loss")
    if acc is None:
        return (float("-inf"), float("inf"))
    return (acc, -(loss if loss is not None else float("inf")))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="取搜索结果里的前 K 组参数")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dir", default="outputs/tuning")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=16, help="精选阶段用的 epoch 数")
    args = parser.parse_args(argv)

    path = Path(args.dir) / f"{args.model}_trials.jsonl"
    if not path.exists():
        raise SystemExit(f"没有找到 {path}")

    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    ok = [r for r in results if r.get("best_val_acc") is not None]
    ok.sort(key=sort_key, reverse=True)
    for rank, res in enumerate(ok[: args.top], 1):
        print(
            f"# rank {rank}: val_acc={res['best_val_acc']:.4f} "
            f"(proxy {res['epochs']} ep)"
        )
        print(json.dumps(res["params"], ensure_ascii=False))


if __name__ == "__main__":
    main()
