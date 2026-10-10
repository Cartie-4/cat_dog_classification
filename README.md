# Catdog_Classification
Catdog_Classfication用于kaggle数据集中猫狗图片分类
## 内容概览
- 'data/':数据存储，包含train和val数据集
- 'src/data':数据读入、dataset datamodule对象构建与预处理
- 'src/models/*':包含作业要求的三种模型，DNN、CNN与RNN
## 快速开始
1) **环境**(python>=3.10, 建议GPU)
   ```bash
   conda create -n catdog python=3.11 -y
   conda activate catdog
   # 按硬件选择合适的 PyTorch 轮子（示例为 CUDA 12.8）
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   # pip install torch --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128
   pip install -r requirements.txt
