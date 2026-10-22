"""汇总所有模型指标，输出一组 markdown 表（供报告/对话使用）。

数据来源全部是仓库里已有的产物：
- outputs/eval/<model>/metrics.json + predictions.csv
- outputs/tuning/<model>_es_search_trials.jsonl（早停式搜索）
- outputs/final_log.txt / final2_log.txt / final2b_log.txt（正式训练逐轮曲线）
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

ROOT = Path(".")
MODELS = ["cnn_attention", "cnn", "rnn", "vit", "dnn"]
LABEL = {
    "cnn_attention": "CNN_attention",
    "cnn": "CNN",
    "rnn": "RNN",
    "vit": "ViT",
    "dnn": "DNN",
}


def load_eval(m):
    p = ROOT / "outputs" / "eval" / m / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_pred(m):
    p = ROOT / "outputs" / "eval" / m / "predictions.csv"
    if not p.exists():
        return {}
    return {r["filename"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def load_curves():
    """解析三个日志，返回 {model: (best_val, train_at_best, epoch, last_epoch)}"""
    out = {}
    for fn in ["final_log.txt", "final2_log.txt", "final2b_log.txt"]:
        p = ROOT / "outputs" / fn
        if not p.exists():
            continue
        cur = None
        best = {}
        last = {}
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            mm = re.match(r"\[[\d:]+\] ---- train (\w+) ----", ln)
            if mm:
                cur = mm.group(1)
                if cur not in best:
                    best[cur] = None
                    last[cur] = 0
                continue
            if ln.startswith("[") and "----" in ln:
                cur = None
                continue
            if not cur:
                continue
            e = re.search(r"Epoch (\d+): 100%.*val_acc=([0-9.]+).*train_acc=([0-9.]+)", ln)
            if e:
                ep, va, ta = int(e.group(1)), float(e.group(2)), float(e.group(3))
                last[cur] = max(last.get(cur, 0), ep)
                if best.get(cur) is None or va > best[cur][0]:
                    best[cur] = (va, ta, ep)
        for k, v in best.items():
            if v is not None:
                out[k] = (v[0], v[1], v[2], last.get(k, v[2]))
    return out


def load_search(m):
    p = ROOT / "outputs" / "tuning" / f"{m}_es_search_trials.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def param_count(m):
    try:
        from catdog.train import build
        model, _ = build(m)
        n = sum(p.numel() for p in model.parameters())
        del model
        return n
    except Exception:
        return None


def cfg_row(m):
    from catdog.train import CONFIG_CLASSES
    c = CONFIG_CLASSES[m]()
    if m == "cnn":
        extra = f"width={c.base_width}"
    elif m == "rnn":
        extra = f"rnn_hidden={c.rnn_hidden}, L={c.num_layers}, patch={c.patch_size}"
    elif m == "cnn_attention":
        extra = f"fnn={c.fnn_hidden}, width={c.base_width}, 变尺寸"
    elif m == "vit":
        extra = f"embed={c.embed_dim}, depth={c.depth}, patch={c.patch_size}, heads={c.num_heads}"
    else:
        extra = f"hidden={c.hidden_dim}, bottleneck={c.bottleneck_dim}"
    return (
        f"lr={c.lr:g}, wd={c.weight_decay:g}, dropout={c.dropout:g}, "
        f"bs={c.data.batch_size}, {extra}"
    )


def main():
    curves = load_curves()
    preds = {m: load_pred(m) for m in MODELS}
    evals = {m: load_eval(m) for m in MODELS}
    searches = {m: load_search(m) for m in MODELS}
    params = {m: param_count(m) for m in MODELS}

    # 错误交集
    files = sorted(set.intersection(*[set(p) for p in preds.values() if p])) if all(preds.values()) else []
    err = {m: {f for f in files if preds[m][f]["correct"] != "1"} for m in MODELS}
    all5 = set.intersection(*err.values()) if files else set()
    noise_floor = (len(files) - len(all5)) / len(files) * 100 if files else float("nan")

    print("## 表 1 · 模型与配置\n")
    try:
        from catdog.data.dataset import CatDogDataset
        n_train = len(CatDogDataset("train"))
    except Exception:
        n_train = 0
    print("| 模型 | 参数量 | 每训练样本参数 (N=%d) | 关键超参 |" % n_train)
    print("|---|---|---|---|")
    for m in MODELS:
        p = params[m]
        per = f"{p/n_train:,.0f}" if n_train else "—"
        print(f"| **{LABEL[m]}** | {p/1e6:.2f} M | {per} | {cfg_row(m)} |")

    print("\n## 表 2 · 早停式搜索（cap 300 / patience 12）\n")
    print("| 模型 | trial 数 | 收敛轮数 中位 | 最好 | 最差 | 跨度 |")
    print("|---|---|---|---|---|---|")
    for m in MODELS:
        rows = [r for r in searches[m] if r.get("best_val_acc")]
        if not rows:
            print(f"| {LABEL[m]} | — | — | — | — | — |")
            continue
        accs = sorted(r["best_val_acc"] * 100 for r in rows)
        eps = sorted(r.get("epochs_run", 0) for r in rows)
        med = eps[len(eps)//2]
        print(f"| {LABEL[m]} | {len(rows)} | {med} | {max(accs):.2f}% | {min(accs):.2f}% | {max(accs)-min(accs):.1f} pt |")

    print("\n## 表 3 · 正式训练与评估（500 张验证集）\n")
    print("| 模型 | 训练 best val_acc | 存档 ckpt 准确率 | ROC-AUC | ckpt 轮次 | 训练轮数 | train_acc@best | 过拟合间隙 |")
    print("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        e = evals[m]
        acc = e["splits"]["val"]["metrics"]["accuracy"] * 100 if e else float("nan")
        auc = e["splits"]["val"]["metrics"]["roc_auc"] if e else float("nan")
        ck = e.get("epoch") if e else None
        cur = curves.get(m)
        if cur:
            bva, bta, bep, last = cur
            print(f"| **{LABEL[m]}** | {bva*100:.2f}% (ep{bep}) | {acc:.2f}% | {auc:.4f} | {ck} | {last+1} | {bta*100:.2f}% | {bta*100-bva*100:+.2f} pt |")
        else:
            print(f"| **{LABEL[m]}** | — | {acc:.2f}% | {auc:.4f} | {ck} | — | — | — |")

    print("\n## 表 4 · 混淆矩阵与逐类指标\n")
    print("| 模型 | 混淆矩阵 [[TP,FP],[FN,TN]] | cat P/R/F1 | dog P/R/F1 |")
    print("|---|---|---|---|")
    for m in MODELS:
        e = evals[m]
        if not e:
            continue
        s = e["splits"]["val"]
        cm = s["confusion_matrix"]
        pc = {x["class"]: x for x in s["metrics"]["per_class"]}
        cat, dog = pc.get("cat", {}), pc.get("dog", {})
        print(f"| {LABEL[m]} | `{cm}` | {cat.get('precision',0):.3f}/{cat.get('recall',0):.3f}/{cat.get('f1',0):.3f} "
              f"| {dog.get('precision',0):.3f}/{dog.get('recall',0):.3f}/{dog.get('f1',0):.3f} |")

    print("\n## 表 5 · 错误构成（五模型联合，上限 ≈ %.2f%%）\n" % noise_floor)
    print("| 模型 | 错分 | 只有它错 | 与其他模型共有 | 五模型全错 | 距上限 |")
    print("|---|---|---|---|---|---|")
    for m in MODELS:
        if not files:
            continue
        uniq = len(err[m] - set.union(*[err[o] for o in MODELS if o != m]))
        e = evals[m]
        acc = e["splits"]["val"]["metrics"]["accuracy"] * 100 if e else 0
        print(f"| {LABEL[m]} | {len(err[m])} | {uniq} | {len(err[m])-uniq} | "
              f"{len(err[m] & all5)} | {noise_floor-acc:.2f} pt |")


if __name__ == "__main__":
    main()
