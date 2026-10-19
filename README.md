# Catdog_Classification
Catdog_Classification 用于 kaggle 数据集中猫狗图片分类。

## 内容概览
- `data/`：数据存储，包含 train 和 val 数据集
- `src/catdog/data`：数据读入、Dataset / DataModule 构建与预处理
- `src/catdog/models/*`：DNN、CNN、RNN 等模型
- `src/catdog/example_data`：预处理的可运行示例

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
   from catdog.data import CatDogDataset, CatDogDataModule
   from catdog.example_data import image_preset, attention_preset
   ```

3) **运行预处理示例**
   ```bash
   python -m catdog.example_data.demo
   # 或（安装后）
   catdog-demo
   ```

## 数据路径
默认读取**当前工作目录**下的 `data/`。也可以用环境变量指定：

```bash
# Windows PowerShell
$env:CATDOG_DATA_DIR = "D:\vscode\cat_dog_classification\data"
```

或在代码中给 `CatDogDataset` / `CatDogDataModule` 显式传 `root=`。
