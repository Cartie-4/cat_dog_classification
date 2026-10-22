# 数据预处理记录

本文档记录 `data/` 数据集的分析结果与预处理决策，供训练（DNN / CNN / RNN 共用）与复现参考。

## 1. 数据集概况

- 总数：**2500** 张（train 2000 / val 500）
- 类别：`cat` / `dog`，各类 1250 张（train 各 1000，val 各 250），**平衡**
- 格式：全部 **RGB**
- 最短边：最小 **50**，中位数 **368**，均值 335（范围 50 → 768）
- 长宽比：0.38 – 2.77，中位数 **1.28**（横图 69% / 竖图 30% / 正方 0.3%）
- 尺寸分布呈双峰：约 375 与约 500 附近，最大 1023×768

### 最短边分布

| 区间 | 张数 | 占比 |
|---|---|---|
| 0–64 | 3 | 0.1% |
| 64–96 | 13 | 0.5% |
| 96–128 | 32 | 1.3% |
| 128–160 | 56 | 2.2% |
| 160–192 | 73 | 2.9% |
| 192–224 | 96 | 3.8% |
| 224–256 | 212 | 8.5% |
| 256–320 | 360 | 14.4% |
| 320–384 | 1164 | 46.6% |
| 384–448 | 319 | 12.8% |
| 448–512 | 171 | 6.8% |
| 512+ | 1 | 0.0% |

## 2. 预处理决策

- **输入尺寸：192×192**
  - 需能被 **3 整除**（192 / 3 = 64），便于 CNN / RNN 共用同一套预处理。
  - 仅 **6.8%** 的图片最短边 < 192，需要放大；大部分为下采样，无插值伪影。
- **小图处理：按短边等比缩放后裁剪**（不使用 padding）
  - 训练：`Resize(192)` → `RandomCrop(192)` → `RandomFlip`
  - 验证：`Resize(192)` → `CenterCrop(192)`
  - 保持长宽比缩小短边，再裁方形，避免黑边引入的分布偏移。
- **剔除规则：最短边 < 96 的图片**（约 192 的一半，且可被 3 整除）
  - 共剔除 **16** 张（train 15 / val 1；dog 9 / cat 7），对类别平衡影响可忽略。
  - 影响很小，但可避免极端小图放大后严重模糊对训练的干扰。

## 3. 被剔除文件清单（最短边 < 96）

| 短边(px) | 文件 | 类别 |
|---|---|---|
| 50 | `data/train/dog.630.jpg` | dog |
| 50 | `data/val/dog.1174.jpg` | dog |
| 59 | `data/train/dog.927.jpg` | dog |
| 72 | `data/train/cat.596.jpg` | cat |
| 74 | `data/train/dog.990.jpg` | dog |
| 75 | `data/train/cat.664.jpg` | cat |
| 80 | `data/train/cat.146.jpg` | cat |
| 81 | `data/train/dog.296.jpg` | dog |
| 89 | `data/train/cat.183.jpg` | cat |
| 89 | `data/train/dog.284.jpg` | dog |
| 89 | `data/train/dog.316.jpg` | dog |
| 90 | `data/train/dog.943.jpg` | dog |
| 93 | `data/train/cat.48.jpg` | cat |
| 94 | `data/train/cat.485.jpg` | cat |
| 95 | `data/train/cat.862.jpg` | cat |
| 95 | `data/train/dog.69.jpg` | dog |

## 4. 代码对应

- 实现位置：`src/catdog/data/`（常量 `DEFAULT_IMAGE_SIZE=192`、`MIN_SHORT_SIDE=96`）
  - `transforms.py`：几何变换、`Normalize`、配置工厂、统计量
  - `cfg.py`：`DataConfig`、`image_preset`
  - `dataset.py`：`CatDogDataset`、变尺寸 collate（补零 + mask）
  - `datamodule.py`：`CatDogDataModule`（LightningDataModule，含统计量缓存）

> 注：早期另有一份自包含示例实现 `src/catdog/example_data/`，它已被删除 ——
> 上面的生产实现就是最终版本，两者内容一致。
>
> 训练/评估口径：训练集按 `MIN_SHORT_SIDE=96` 过滤（2000 → 1985 张）；
> **验证集不做尺寸过滤**（2000 之外的 500 张全部参与评估），见
> `CatDogDataModule(train_min_short_side=..., val_min_short_side=0)`。
