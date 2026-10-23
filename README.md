# Catdog_Classification
Catdog_Classification 用于 kaggle 数据集中猫狗图片分类。

## 内容概览
- `data/`：数据存储，包含 train 和 val 数据集
- `src/catdog/data`：数据读入、Dataset / DataModule 构建与预处理
- `src/catdog/models/*`：DNN、CNN、RNN、CNN_attention、ViT 等模型
- `src/catdog/tune.py`：超参搜索；`src/catdog/evaluate.py`：checkpoint → 混淆矩阵
- `tools/`：分析工具（搜索结果汇总、错误对比、学习曲线等，见下文）
- `docs/`：调参报告（Word）、模型结构图（HTML）

## 快速开始
1) **环境**（python>=3.10，建议 GPU）
   ```bash
   conda create -n catdog python=3.11 -y
   conda activate catdog
   # 按硬件选择合适的 PyTorch 轮子（示例为 CUDA 12.8）
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   # pip install torch --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128
   pip install -r requirements.txt
   ```

2) **以可编辑模式安装本库**
   ```bash
   pip install -e .
   ```
   安装后可在任意目录导入：
   ```python
   from catdog.data import CatDogDataset, CatDogDataModule, image_preset
   ```

3) **训练一个模型**（见下方「训练」）
   ```bash
   python -m catdog.train --model cnn
   ```

## 数据路径
默认读取**当前工作目录**下的 `data/`。也可以用环境变量指定：

```bash
# Windows PowerShell
$env:CATDOG_DATA_DIR = "D:\vscode\cat_dog_classification\data"
```

或在代码中给 `CatDogDataset` / `CatDogDataModule` 显式传 `root=`。

## 预处理

两套管线，见 `src/catdog/data/cfg.py`：

| 管线 | 适用模型 | 流程 |
|---|---|---|
| `image_preset()` | DNN / CNN / RNN / ViT | 短边缩放 192 → 随机裁剪 192（验证中心裁剪）→ 水平翻转 → Normalize → **随机旋转 ±15°（仅训练）** |
| `_attention_data()` | CNN_attention | 保留原图尺寸 → 翻转（训练再加旋转）→ Normalize → 变尺寸 collate 补零 + mask |

> 旋转放在 `Normalize` **之后**：这样补出来的边界值是 0，而 0 在标准化空间里正好是逐通道均值色，
> 不会在图像外圈留黑边。注意随机旋转**对 RNN 是负收益**（−2.4 pt）、对 CNN 是 +2.0 pt，
> 换模型前最好先确认一下。

## 训练
```bash
python -m catdog.train --model cnn --epochs 30          # dnn / cnn / rnn / cnn_attention / vit
python -m catdog.train --model cnn --fast-dev-run       # 1 个 batch 冒烟测试
```

**存档规则很重要**：`--monitor` 决定 checkpoint 按哪个指标挑最优轮次。

```bash
python -m catdog.train --model cnn                    # 默认 val_loss（老行为）
python -m catdog.train --model cnn --monitor val_acc   # 按验证准确率存档（推荐）
```

`val_loss` 的最低点和 `val_acc` 的峰值经常不在同一轮，按准确率存档能让五个模型平均 **+1.3 pt**，
且不需要重训。**但早停始终用 `val_loss`**（`val_acc` 在训练早期抖动 ±2 pt，会让早停提前触发）。

每次运行在 `outputs/<日期_时间>/` 下保存 checkpoint 与 TensorBoard 日志。
不传 `--epochs` 时使用该模型 `cfg.py` 里的 `max_epochs`（默认 300，实际由 EarlyStopping 终止）。

## 当前成绩

500 张验证集，全部用 `--monitor val_acc` 存档：

| 模型 | 准确率 | ROC-AUC | 参数量 | 存档轮次 |
|---|---|---|---|---|
| CNN_attention | **90.20%** | 0.9640 | 11.77 M | 57 |
| CNN | **90.00%** | 0.9599 | 11.18 M | 86 |
| ViT | 73.60% | 0.7960 | 1.70 M | 34 |
| RNN | 71.40% | 0.7513 | 0.53 M | 34 |
| DNN | 64.20% | 0.6846 | 56.76 M | 8 |

详细分析见 `docs/调参报告.docx`，结构图见 `docs/models_architecture.html`。

## 超参数调优
`catdog.tune` 做**随机搜索**，结果增量写入 `outputs/tuning/<model>_<tag>_trials.jsonl`。

```bash
# 代理预算：固定 8 epoch（适合快速粗筛，注意：短预算会把 lr 排序弄反）
python -m catdog.tune --model cnn --trials 10 --epochs 8

# 推荐做法：每组超参都训练到自己的早停点，比"各自收敛后的精度"
#   --early-stop-patience 12 → EarlyStopping(val_loss, patience=12)
#   --epochs 300 只是上限；--progress-every 10 每 10 轮打印一次，方便盯平台期
python -m catdog.tune --model cnn --trials 12 --epochs 300 \
    --early-stop-patience 12 --progress-every 10 --tag rot_search

# 基线：完全不覆盖超参
python -m catdog.tune --model cnn --pin '{}' --tag baseline

# 复跑某一组具体参数（Windows 建议用 --params-file，避免命令行引号被吞）
python -m catdog.tune --model cnn --params-file best.json --save-ckpt --tag final
```

搜索时若要顺手存下最优权重，用 `--save-ckpt` 配合 `--ckpt-monitor {val_loss,val_acc}`
（默认 `val_loss`，与 `catdog.train` 的 `--monitor` 含义一致）。

**关键经验：lr 与训练轮数强耦合。** 同一个 lr 在 8 轮预算下可能垫底、在 90 轮预算下
可能夺冠（本项目 CNN 就是这样：lr 8.3e-3 在 8 轮时只有 67.5%，给足 91 轮后到 87.6%）。
所以比较超参一定要**训练到早停**，或者至少确认验证曲线已经进入平台期。

## 验证与混淆矩阵
`catdog.evaluate` 从 checkpoint 还原模型（结构/超参自动从 ckpt 推断），在验证集上
输出混淆矩阵与分类指标。

```bash
python -m catdog.evaluate --ckpt outputs/final3/cnn_acc/<时间>/checkpoints --split val
python -m catdog.evaluate --ckpt <ckpt 文件> --split both --top-k-errors 20
```

产物（默认 `outputs/eval/<ckpt名>_<split>/`）：

| 文件 | 内容 |
|---|---|
| `confusion_matrix.png` | 计数 + 按真实类归一化的混淆矩阵热力图（Pillow 绘制，不依赖 matplotlib） |
| `confusion_matrix.txt` | 纯文本混淆矩阵 |
| `metrics.json` | accuracy / 逐类 precision-recall-F1 / ROC-AUC（正类 = dog）/ 平均置信度 |
| `predictions.csv` | 逐样本真实标签、预测、各类概率 |

> 说明：`--split train` 用**确定性验证管线**跑训练集，避免随机增强影响指标。

## 权重文件在哪

> **重要：`outputs/` 已在 `.gitignore` 里，所以下面这些文件只在本地工作目录，不在版本库中**
> （`git ls-files outputs` 返回空）。换机器/重新 clone 后不会带过来，需要重跑训练自己生成。

五个模型最终权重的确切路径（共约 313 MB，`last.ckpt` 是各目录里的另一份）：

| 模型 | ckpt 路径 | 大小 |
|---|---|---|
| CNN_attention | `outputs/final3/cnn_attention_acc/2026-10-23_09-46-47/checkpoints/epoch=57-val_acc=0.9020.ckpt` | 45.0 MB |
| CNN | `outputs/final3/cnn_acc/2026-10-23_08-33-15/checkpoints/CNN-epoch=86-val_acc=0.9000.ckpt` | 42.7 MB |
| ViT | `outputs/final3/vit_acc/2026-10-23_09-15-42/checkpoints/ViT-epoch=34-val_acc=0.7360.ckpt` | 6.5 MB |
| RNN | `outputs/final3/rnn_acc/2026-10-23_09-31-50/checkpoints/RNN_epoch=34-val_acc=0.7140.ckpt` | 2.0 MB |
| DNN | `outputs/final3/dnn_acc/2026-10-23_09-10-44/checkpoints/DNN-epoch=08-val_acc=0.6420.ckpt` | 216.5 MB |

想直接从权重回到混淆矩阵：

```bash
python -m catdog.evaluate --ckpt outputs/final3/cnn_acc/<时间>/checkpoints --out outputs/eval3/cnn_acc
```

**现成的五个混淆矩阵**（对应上表权重的评估结果，同样只在本地）：

| 模型 | 图 | 数据 |
|---|---|---|
| CNN_attention | `outputs/eval3/cnn_attention_acc/confusion_matrix.png` | 同目录 `metrics.json` / `predictions.csv` |
| CNN | `outputs/eval3/cnn_acc/confusion_matrix.png` | 同上 |
| ViT | `outputs/eval3/vit_acc/confusion_matrix.png` | 同上 |
| RNN | `outputs/eval3/rnn_acc/confusion_matrix.png` | 同上 |
| DNN | `outputs/eval3/dnn_acc/confusion_matrix.png` | 同上 |

> 注意：上表这几张图在 `outputs/` 下，**同样不在版本库里**。
> 真正进了版本库的只有 `docs/` 下的内容——包括拼图版 `docs/confusion_matrices.png`、
> `docs/models_architecture.html`、`docs/调参报告.docx` 等。

五张拼成一张的总览图在 `docs/confusion_matrices.png`；
`docs/models_architecture.html` 的「混淆矩阵」一节也直接嵌了这五张图。

## 分析工具（`tools/`）

调参和分析过程中沉淀下来的独立脚本，都可以单独运行、不需要改库代码。

| 工具 | 作用 |
|---|---|
| `tools/all_metrics.py` | **总表生成器**。把五个模型的参数量、超参、搜索结果、最终训练/评估、混淆矩阵、错误构成汇总成多张 markdown 表（输出到 stdout，可重定向存文件）。`--eval-dir`/`--suffix` 指定评估目录与搜索标签；定位模型目录时会自动尝试 `<model>/` 和 `<model>_*/`。开完一轮调参后跑它就能拿到全部数字，本工作目录 `outputs/all_metrics.md` 就是它的产物（未进版本库）。 |
| `tools/summarize_tuning.py` | **搜索结果汇总**。读 `<model>_trials.jsonl`，输出"基线 vs 搜索最优"的对比表与逐 epoch 曲线，用于快速看某个模型的搜索全景。 |
| `tools/analyze_search.py` | **早停式搜索分析**。除了排名，还输出每组的**实际收敛轮数**、峰值出现在第几轮、`lr ↔ 收敛轮数`对照表、以及曲线尾部（判断是否真的进平台期）。专门用来回答"搜几个 epoch 才够"。 |
| `tools/compare_errors.py` | **两个模型的错分集合对比**。回答"这两个模型错的是同一批图吗"：错分交集、Jaccard、独立性假设下的期望交集 + 超几何检验、按真实类别拆分、共同错分样本清单。也用来估"与模型无关的上限"（所有模型都错的图 ≈ 标签噪声）。 |
| `tools/data_scaling.py` | **数据量标定（学习曲线 + 幂律拟合）**。按比例分层抽样训练集（保持猫狗 1:1），在每个子集上训练，拟合 `acc(N) = a − b·N^(−α)`，输出：外推天花板 `a`、数据充足度 `acc(N)/a`、数据缺口、以及"达到目标精度还需要多少张图"。用于定量回答"这个模型是缺数据还是缺归纳偏置"。 |
| `tools/top_params.py` | **取 TOP-K 参数**。从搜索结果里直接导出前 K 组参数的 JSON，配合 `catdog.tune --params-file` 做第二轮复核。 |
| `tools/attention_diag.py` | **CNN_attention 专项诊断**。测变尺寸 batch 的补零占比；用 `--data {attn,img192}` 切换"原图变尺寸"与"192+随机裁剪"预处理；用 `--head {attn,gap}` 把注意力换成平均池化做对照。 |

常用组合：

```bash
# 一轮调参结束后的标准动作
python tools/analyze_search.py --dir outputs/tuning --suffix rot_search --out search_summary.md
python -m catdog.evaluate --ckpt outputs/final3/cnn_acc/<时间>/checkpoints --out outputs/eval3/cnn_acc
python tools/all_metrics.py --eval-dir outputs/eval3 --suffix rot_search > outputs/all_metrics.md

# 判断两个模型是不是错在同一批图上
python tools/compare_errors.py \
    --a outputs/eval3/cnn_acc/predictions.csv cnn \
    --b outputs/eval3/vit_acc/predictions.csv vit \
    --out outputs/eval3/error_overlap_cnn_vit.json

# 某个模型到底缺不缺数据（三个规模 + 幂律外推，约 0.5~2 小时/模型）
python tools/data_scaling.py --model vit --fractions 0.25 0.5 1.0 --epochs 40
```

## 目录产物约定

```
outputs/
├── final3/<model>_acc/<日期_时间>/checkpoints/  # 正式训练（按 val_acc 存档）的最优 + last ckpt
├── eval3/<model>_acc/                           # 该 ckpt 的混淆矩阵、指标、逐样本预测
├── tuning/<model>_<tag>_trials.jsonl            # 超参搜索的原始记录（每组一行，含 epochs_run）
├── tuning/<model>_<tag>_best.json               # 搜索最优参数，可直接喂给 --params-file
├── all_metrics.md                               # all_metrics.py 生成的总表
├── cfg_backup/                                  # 改 cfg 之前的备份
└── round3_final_log.txt / rot_search_log.txt    # 训练与搜索的原始日志
```

`outputs/` 已在 `.gitignore` 中，不会进版本库。

> 注意：`all_metrics.py` 解析模型目录时依次尝试 `<model>/`、`<model>_*/`，
> 所以只保留 `*_acc` 目录（删掉旧的 `val_loss` 存档）也能直接跑。

